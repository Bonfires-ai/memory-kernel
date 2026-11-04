"""Answer-context assembly over ranked search hits. Pure: plain dicts in, text out.

The Neo4j reader (``kg_reader``) supplies the windows, entity sheets and timeline rows; nothing here touches a driver.

Layers, filled in budget order: (a) the entity sheet, (b) conversation windows around the ranked messages, (c) a
timeline, only for the shapes in ``ContextConfig.timeline_shapes``. The sheet takes at most ``sheet_share`` of the
token budget and the timeline at most ``timeline_share`` (the timeline's need is reserved before the windows fill).
Tokens are counted with tiktoken ``cl100k_base``. There is no fallback: a word-count proxy would change the contexts a
build was benchmarked with, so an encoder that cannot load is a refusal (``TokenizerUnavailable``) that names the fix
(``mk setup --component tiktoken``). The encoder loads on the first count, never at import.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from memory_kernel.graph.envelope import MODALITY, force_markers
from memory_kernel.graph.kg_contract import RESOLVED_TIME_SOURCES, ext
from memory_kernel.search.kg_types import MessageHit
from memory_kernel.search.summary_lines import SheetLine, note_missing

ENCODING = "cl100k_base"


class TokenizerUnavailable(ValueError):
    """The token encoder cannot be loaded (a refusal, never a traceback, never a silent proxy)."""


_ENC: Any = None
#: Assemblies run in worker threads (``kg_reader.build_answer_context``): one thread loads the encoder, and
#: ``_load_encoder``'s swap of ``TIKTOKEN_CACHE_DIR`` is never undone under another thread's load.
_ENC_LOCK = threading.Lock()


def _cache_dirs() -> list[Path]:
    """Where the BPE file may live, in order: ``TIKTOKEN_CACHE_DIR``, then ``<data>/tiktoken`` (``mk setup``'s place;
    ``<data>`` is ``MK_DATA_DIR`` or the user data dir). tiktoken's own default cache is the last resort (the empty
    answer here)."""
    from memory_kernel import setup_runtime

    out = [Path(env)] if (env := os.environ.get("TIKTOKEN_CACHE_DIR")) else []
    return [*out, setup_runtime.resolve_data_dir(None, os.environ) / "tiktoken"]


def _load_encoder() -> Any:
    import hashlib

    import tiktoken

    from memory_kernel import setup_runtime

    url = setup_runtime.load_pins()["components"]["tiktoken"]["url"]
    key = hashlib.sha1(url.encode()).hexdigest()  # tiktoken's cache file name
    holder = next((d for d in _cache_dirs() if (d / key).is_file()), None)
    prior = os.environ.get("TIKTOKEN_CACHE_DIR")
    if holder is not None:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(holder)
    try:
        return tiktoken.get_encoding(ENCODING)
    except Exception as exc:  # noqa: BLE001 -- offline, corrupt file, missing package: one refusal
        raise TokenizerUnavailable(
            f"the {ENCODING} token encoder cannot be loaded ({type(exc).__name__}: {exc}): run "
            "`mk setup --component tiktoken` (or set TIKTOKEN_CACHE_DIR to a directory that holds it)"
        ) from None
    finally:
        if holder is not None:
            if prior is None:
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)
            else:
                os.environ["TIKTOKEN_CACHE_DIR"] = prior


def count_tokens(text: str) -> int:
    """Deterministic token count (tiktoken ``cl100k_base``); ``TokenizerUnavailable`` when the encoder cannot load."""
    global _ENC
    if not text:
        return 0
    if _ENC is None:
        with _ENC_LOCK:
            if _ENC is None:
                _ENC = _load_encoder()
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class ContextConfig:
    """Every parameter of the answer-context assembly."""

    window_before: int = 2
    window_after: int = 2
    per_saga_cap: int = 2  # disjoint windows per saga (overlapping ones merge and count once)
    token_budget: int = 3500
    sheet_share: float = 0.30  # max share of the budget for the entity sheet
    timeline_share: float = 0.20  # max share of the budget for the timeline
    entity_sheet_k: int = 3
    sheet_lines_cap: int = 12
    identity_lines_cap: int = 3  # identity lines (copular nominal statements about the entity) kept before the rest
    timeline_shapes: frozenset[str] = frozenset({"when", "how_long"})
    timeline_cap: int = 15
    evidence_k: int = 40  # >0: the top-k hits' message lines in date order, before the windows (frozen: K=40)
    evidence_share: float = 0.5  # max share of the budget for the evidence section
    relative_time: bool = True  # annotate relative dates in message lines ("yesterday" -> its resolved day)


def _date(value: object) -> str:
    return str(value or "")[:10]


def _session(msg: Mapping[str, Any]) -> str:
    """``S<n>`` from the saga name, else the episode's ``source_description``; empty when neither names a session."""
    raw = str(msg.get("session") or msg.get("session_id") or ext(msg, "source_description") or "")
    digits = re.findall(r"\d+", raw)
    return f"S{digits[-1]}" if digits else (f"S{raw}" if raw else "")


def _text(msg: Mapping[str, Any]) -> str:
    if msg.get("text"):
        return str(msg["text"])
    content = str(msg.get("content") or "")
    sender = str(ext(msg, "sender_name") or "")
    return content[len(sender) + 1 :].strip() if sender and content.startswith(f"{sender}:") else content


def _clock(msg: Mapping[str, Any]) -> Any:
    return ext(msg, "told_at") or msg.get("valid_at")


def _relative_text(msg: Mapping[str, Any]) -> str:
    from memory_kernel.knowledge.temporal_deixis import content_with_relative_time

    return content_with_relative_time(_text(msg), _clock(msg))


def relative_texts(messages: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """``{message uuid: its text with its resolved-relative-time note}``, the text ``message_line(relative_time=True)``
    shows, for ``messages`` (uuid -> message row). A bare weekday's note reads the sentence's tense from a parse of the
    message, so serve computes this once per message when a run opens (``Server``) and hands it to every question's
    assembly: a question never parses a message."""
    return {str(u): _relative_text(m) for u, m in messages.items()}


def message_line(
    msg: Mapping[str, Any], *, relative_time: bool = False, relative: Mapping[str, str] | None = None
) -> str:
    """``[S<session> · <YYYY-MM-DD> · Sender → Addressee] text``. A part the row does not carry (no session, no
    sender, no addressee) is omitted, never replaced by a placeholder. With ``relative_time``, the text carries its
    resolved-relative-time note: read from ``relative`` (``relative_texts``, by message uuid) when it holds the message,
    else computed here."""
    who = str(ext(msg, "sender_name") or "")
    if who and ext(msg, "addressee_name"):
        who += f" → {ext(msg, 'addressee_name')}"
    clock = _clock(msg)
    head = " · ".join(p for p in (_session(msg), _date(clock), who) if p)
    text = _text(msg)
    if relative_time:
        known = (relative or {}).get(str(msg.get("uuid")))
        text = known if known is not None else _relative_text(msg)
    return f"[{head}] {text}".rstrip() if head else text.rstrip()


def _marked(text: str, item: Mapping[str, Any]) -> str:
    return " ".join([text, *force_markers({"asserted": ext(item, "asserted"), "modality": ext(item, "modality")})])


def _support_line(item: Mapping[str, Any]) -> tuple[str, str]:
    """(dedup key, rendered line) of one supporting fact or event."""
    if item.get("kind") == "event":
        name = str(item.get("name") or item.get("fact") or "")
        line = f"↳ event: {name}"
        if item.get("happened_at"):
            line += f" · happened {_date(item['happened_at'])}"
        return f"event|{name.lower()}", _marked(line, item)
    fact = str(item.get("fact") or item.get("name") or "")
    return f"fact|{fact.lower()}", _marked(f"↳ fact: {fact}", item)


def _merge_windows(hits: Sequence[MessageHit], windows: Mapping[str, Mapping[str, Any]], cap: int) -> list[dict]:
    """Windows in hit-score order; overlapping windows of one saga merge; at most ``cap`` disjoint per saga."""
    merged: list[dict] = []
    for h in hits:
        w = windows.get(h.message)
        if not w:
            continue
        msgs = {m["uuid"]: m for m in w["messages"]}
        saga = w.get("saga")
        same = [x for x in merged if x["saga"] == saga]
        overlap = [x for x in same if msgs.keys() & x["msgs"].keys()]
        if not overlap and saga is not None and len(same) >= cap:  # saga-less windows have no saga to cap on
            continue
        if not overlap:
            merged.append({"saga": saga, "anchor": h.message, "msgs": dict(msgs)})
            continue
        into = overlap[0]
        for other in overlap[1:]:
            into["msgs"].update({k: v for k, v in other["msgs"].items() if k not in into["msgs"]})
            merged.remove(other)
        for uid, m in msgs.items():
            if uid not in into["msgs"]:
                into["msgs"][uid] = m
            else:  # same message in both windows: union the support rows
                known = into["msgs"][uid]
                known["support"] = [*known.get("support", ()), *m.get("support", ())]
    return merged


def _ordered(window: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    indexed = list(enumerate(window["msgs"].values()))
    indexed.sort(key=lambda p: (str(p[1].get("valid_at") or ext(p[1], "told_at") or ""), p[0]))
    return [m for _, m in indexed]


def _block(
    msg: Mapping[str, Any],
    seen: set[str],
    *,
    relative_time: bool = False,
    relative: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """The lines of one message: its line, then its support lines not already seen. Registers keys only if kept."""
    lines = [("msg|" + str(msg["uuid"]), message_line(msg, relative_time=relative_time, relative=relative))]
    local: set[str] = set()
    for item in msg.get("support") or ():
        key, line = _support_line(item)
        if key in seen or key in local:
            continue
        local.add(key)
        lines.append((key, line))
    return lines


_LINE_HEAD = re.compile(r"\[([^\]]*)\]\s*(.*)", re.DOTALL)
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_SESSION = re.compile(r"S\d+")
#: The force markers a summary line may end with (rendered by ``envelope.force_markers``): not a plain statement.
_MARKERS = tuple(
    dict.fromkeys(mk for m in MODALITY.values() for mk in force_markers({"asserted": False, "modality": m}))
)


def _speaker_and_statement(line: str) -> tuple[str, str]:
    """(speaker, statement) of a summary line ``[sender, date] fact`` / ``[sender] fact`` / ``[date] fact``, or a
    message line ``[S<n> · date · Sender → Addressee] text``. The speaker is empty when the head names none."""
    m = _LINE_HEAD.fullmatch(line.strip())
    if not m:
        return "", line.strip()
    head, statement = m.group(1).strip(), m.group(2).strip()
    if " · " in head:
        last = head.split(" · ")[-1].strip()  # the sender comes last; a session- or date-only head names none
        return ("" if _DAY.fullmatch(last) or _SESSION.fullmatch(last) else last.split(" → ")[0].strip()), statement
    sender, _, tail = head.rpartition(", ")
    if sender and _DAY.fullmatch(tail):
        return sender.strip(), statement
    return ("" if _DAY.fullmatch(head) or _SESSION.fullmatch(head) else head), statement


def _name_span(head: Any) -> str:
    """The subject head's own name: the head with its ``compound`` / ``flat`` children (recursively), as the doc's text
    between the first and last of them (original whitespace, so "Jean-Luc" stays whole). Possessors (``poss``),
    determiners and other modifiers are not the subject's name: "Kira Dace's brother" names the brother."""
    toks, todo = [head], [head]
    while todo:
        more = [c for c in todo.pop().children if c.dep_ in ("compound", "flat")]
        toks += more
        todo += more
    return head.doc[min(t.i for t in toks) : max(t.i for t in toks) + 1].text


def _names_entity(text: str, name: str) -> bool:
    from memory_kernel.graph.identity import alias_occurs

    return alias_occurs(name, text, is_common=lambda _: False)  # whole word, any case


@lru_cache(maxsize=4096)
def identity_line(line: str, name: str) -> bool:
    """True when the line's statement is a copular clause with a nominal predicate whose subject is the entity.

    spaCy: a ROOT with lemma "be" and an ``attr`` child that is a NOUN/PROPN, or a NOUN/PROPN ROOT with a ``cop``
    child; its ``nsubj`` head's own name span (``_name_span``) names the entity (whole word, any case), or is a 1st-person singular pronoun when the
    line's speaker is the entity. A line ending in a force marker ("(asked)", ...) is not a plain statement."""
    speaker, statement = _speaker_and_statement(line)
    if not statement or not name or statement.endswith(_MARKERS):
        return False
    own = speaker.casefold() == name.casefold()
    if not own and not _names_entity(statement, name):  # cheap gate: the parse could not find the entity
        return False
    from memory_kernel.graph.g1_nlp import query_doc

    # a model that will not load reads on sm (with a note) instead of raising: /context never goes empty over it
    doc = query_doc(statement)
    for sent in doc.sents if doc is not None else ():
        root = sent.root
        copular = (
            root.lemma_ == "be" and any(c.dep_ == "attr" and c.pos_ in ("NOUN", "PROPN") for c in root.children)
        ) or (root.pos_ in ("NOUN", "PROPN") and any(c.dep_ == "cop" for c in root.children))
        if not copular:
            continue
        for subj in (c for c in root.children if c.dep_ == "nsubj"):
            if _names_entity(_name_span(subj), name):
                return True
            first_person = (
                subj.pos_ == "PRON" and subj.morph.get("Person") == ["1"] and subj.morph.get("Number") == ["Sing"]
            )
            if own and first_person:
                return True
    return False


def _recent(lines: list[str], cap: int) -> list[str]:
    return lines[-cap:] if cap > 0 else []


def _sheet(entities: Sequence[Mapping[str, Any]], config: ContextConfig, budget: int, seen: set[str]) -> list[str]:
    """Per entity: its identity lines first (``identity_lines_cap``, original order), then its most recent lines up to
    ``sheet_lines_cap`` in all, chronologically. Identity lines take the room first; recent lines win what is left."""
    out: list[str] = []
    used = count_tokens("## Entity sheet") + 1
    for ent in entities[: config.entity_sheet_k]:
        recs = ent.get("lines")
        if recs is None:  # a row built before 1.4.1: recency only, nothing parsed
            note_missing(str(ent.get("group_id") or ""))
            recs = [SheetLine("", ln, False) for ln in str(ent.get("summary") or "").split("\n")]
        # a line's embedded newlines render as spaces; the stored text is unchanged
        shown = [(" ".join(r.text.split("\n")).strip(), r.identity) for r in recs]
        lines = list(dict.fromkeys(ln for ln, _ in shown if ln and ln not in seen))
        name = str(ent.get("name") or "?")
        ident: list[str] = []
        if config.identity_lines_cap > 0:
            cap = min(config.identity_lines_cap, config.sheet_lines_cap)
            marked = {ln for ln, is_id in shown if is_id}
            ident = [ln for ln in lines if ln in marked][:cap]
            lines = [ln for ln in lines if ln not in ident]  # identity lines past the cap rejoin the recent pool
        lines.sort(key=lambda ln: _line_date(ln))  # stable: most recent last
        lines = _recent(lines, config.sheet_lines_cap - len(ident)) if ident else lines[-config.sheet_lines_cap :]
        used_head = used + count_tokens(name) + 1
        if used_head > budget:
            break
        first: list[str] = []
        used = used_head
        for ln in ident:  # identity lines take the room first; one that does not fit leaves it to the next
            cost = count_tokens(ln) + 1
            if used + cost > budget:
                continue
            first.append(ln)
            used += cost
        kept: list[str] = []
        for ln in reversed(lines):  # recent lines win the room
            cost = count_tokens(ln) + 1
            if used + cost > budget:
                break
            kept.append(ln)
            used += cost
        kept.reverse()
        kept = [*first, *kept]
        if not kept:  # a bare name tells the answerer nothing
            break
        out.append(name)
        out.extend(kept)
        seen.update(kept)
    return out


def _evidence(
    hits: Sequence[MessageHit],
    windows: Mapping[str, Mapping[str, Any]],
    config: ContextConfig,
    room: int,
    relative: Mapping[str, str] | None = None,
) -> list[str]:
    """Message lines of the top ``evidence_k`` hits (those with a window), chronological; as many as fit ``room``."""
    picked = []
    for h in hits:
        if len(picked) >= config.evidence_k:
            break
        w = windows.get(h.message)
        own = next((m for m in (w or {}).get("messages", ()) if m["uuid"] == h.message), None)
        if own is not None:
            picked.append(own)
    picked.sort(key=lambda m: (str(ext(m, "told_at") or m.get("valid_at") or ""), str(m["uuid"])))
    out: list[str] = []
    used = count_tokens("## Evidence") + 1
    for m in picked:
        line = message_line(m, relative_time=config.relative_time, relative=relative)
        cost = count_tokens(line) + 1
        if used + cost > room:
            break
        out.append(line)
        used += cost
    return out


def _line_date(line: str) -> str:
    m = re.match(r"\[[^\]]*?(\d{4}-\d{2}-\d{2})[^\]]*\]", line)
    return m.group(1) if m else ""


def _timeline(timeline: Sequence[Mapping[str, Any]], config: ContextConfig) -> list[tuple[str, str]]:
    """Dated Events. ``happened_at`` is shown as a date only when its ``time_source`` resolved it (deictic or
    explicit); an unresolved or told-default Event shows what the speaker said (the original time text when the row
    carries one) and the day it was said: ``[a few weeks ago] ... (said 2023-06-27)``."""
    dated = []
    for ev in timeline:
        happened, told = ext(ev, "happened_at"), ext(ev, "told_at")
        name = str(ext(ev, "source_text") or ev.get("name") or ev.get("fact") or "")
        if happened and ext(ev, "time_source") in RESOLVED_TIME_SOURCES:
            when = str(happened)
            line = f"[{_date(happened)}] {name}"
            if told and _date(told) != _date(happened):
                line += f" (told {_date(told)})"
        elif told:
            when = str(told)
            text = ext(ev, "time_text") or ext(ev, "argm_tmp_text")
            line = f"[{text}] {name} (said {_date(told)})" if text else f"[said {_date(told)}] {name}"
        else:
            continue
        dated.append((when, name, f"event|{name.lower()}", _marked(line, ev)))
    dated.sort(key=lambda t: (t[0], t[1]))
    return [(key, line) for _, _, key, line in dated[: config.timeline_cap]]


def _fill_window(blocks: list[list[tuple[str, str]]], anchor: int, room: int, first: bool) -> list[int] | None:
    """Indexes of blocks admitted into ``room`` tokens. A whole window or (the first one only) its nearest blocks."""
    costs = [sum(count_tokens(ln) + 1 for _, ln in b) for b in blocks]
    if sum(costs) <= room:
        return list(range(len(blocks)))
    if not first:
        return None
    chosen, used = [], 0
    for i in sorted(range(len(blocks)), key=lambda i: (abs(i - anchor), i)):
        if used + costs[i] > room:
            break
        chosen.append(i)
        used += costs[i]
    return sorted(chosen) or None


def assemble_context(
    ranked_messages: Sequence[MessageHit],
    *,
    windows: Mapping[str, Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
    shape: str,
    config: ContextConfig,
    relative: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Layered answer context: entity sheet, then conversation windows, then timeline (for timeline shapes only).

    ``windows`` maps a ranked message uuid to ``{"saga", "messages": [...]}``, each message a dict with ``uuid``,
    ``message_id``, ``sender_name``, ``addressee_name``, ``valid_at``/``told_at``, ``session``, ``content``/``text``
    and optional ``support`` rows (``kind`` event/fact, ``fact``/``name``, ``happened_at``, ``asserted``,
    ``modality``). ``relative``: the messages' relative-time texts computed ahead (``relative_texts``; serve's, at
    open), read instead of computed. Returns ``{"text", "sections", "message_ids", "tokens"}``; a pure function of its
    inputs.
    """
    budget = config.token_budget
    seen: set[str] = set()
    sections: dict[str, str] = {}

    sheet = _sheet(entities, config, int(budget * config.sheet_share), seen)
    if sheet:
        sections["entities"] = "\n".join(["## Entity sheet", *sheet])
    sheet_tokens = count_tokens(sections.get("entities", "")) + (1 if sheet else 0)

    tl_lines = _timeline(timeline, config) if shape in config.timeline_shapes else []
    tl_cost = lambda ls: count_tokens("## Timeline") + 1 + sum(count_tokens(ln) + 1 for _, ln in ls)  # noqa: E731
    tl_room = int(budget * config.timeline_share)
    reserve = min(tl_cost(tl_lines), tl_room) if tl_lines else 0

    evidence = _evidence(ranked_messages, windows, config, int(budget * config.evidence_share), relative)
    if evidence:
        sections["evidence"] = "\n".join(["## Evidence", *evidence])
    evidence_tokens = count_tokens(sections.get("evidence", "")) + (1 if evidence else 0)

    win_room = budget - sheet_tokens - evidence_tokens - reserve - count_tokens("## Conversation") - 1
    win_lines: list[str] = []
    ids: list[str] = []
    first = True
    for w in _merge_windows(ranked_messages, windows, config.per_saga_cap):
        ordered = _ordered(w)
        blocks = [_block(m, seen, relative_time=config.relative_time, relative=relative) for m in ordered]
        anchor = next((i for i, m in enumerate(ordered) if m["uuid"] == w["anchor"]), 0)
        chosen = _fill_window(blocks, anchor, win_room, first)
        if chosen is None:
            break
        first = False
        for i in chosen:
            for key, line in blocks[i]:
                win_lines.append(line)
                seen.add(key)
                win_room -= count_tokens(line) + 1
            ids.append(str(ordered[i].get("message_id") or ordered[i]["uuid"]))
    if win_lines:
        sections["windows"] = "\n".join(["## Conversation", *win_lines])

    if tl_lines:  # dedup against the event lines the windows actually emitted, then fit the timeline's room
        room = budget - sheet_tokens - (count_tokens(sections.get("windows", "")) + 1 if win_lines else 0)
        room = min(room, tl_room)
        kept: list[str] = []
        used = count_tokens("## Timeline") + 1
        for key, line in tl_lines:
            if key in seen or line in seen:
                continue
            cost = count_tokens(line) + 1
            if used + cost > room:
                break
            kept.append(line)
            used += cost
            seen.add(key)
        if kept:
            sections["timeline"] = "\n".join(["## Timeline", *kept])

    text = "\n\n".join(sections.values())
    return {"text": text, "sections": sections, "message_ids": ids, "tokens": count_tokens(text)}
