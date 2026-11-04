"""Shared spaCy loader and filler-analysis helpers for the G1 projection and its modules.

A leaf module (no graph imports): ``g1_projection``, ``g1_referring`` and ``g1_render`` all import from here, so none of
them imports another through a cycle. ``g1_projection`` re-exports these names for existing importers.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

#: Leading parts of speech stripped from a filler to get its surface.
STRIP_LEAD = frozenset({"ADP", "DET"})
#: Head parts of speech of a referring noun phrase.
REFERRING_HEAD = frozenset({"NOUN", "PROPN"})
#: Why a parse is wanted. ``extraction`` is the build's (fail loud: a silent fallback would build a different graph),
#: ``query`` is serve's (never fails: a missing model degrades to the small one), ``turn`` is the runtime's (same).
Purpose = Literal["turn", "extraction", "query"]
#: Set (as a ``parser_note``) when a purpose reads on the fallback model because the configured one would not load.
PARSER_FALLBACK = "sm-fallback"
#: ``MK_SPACY_GPU=1`` asks spaCy for a GPU (``spacy.prefer_gpu()``, needs ``memory-kernel[gpu]``) before the first model
#: loads; with no usable GPU spaCy stays on the CPU.
GPU_ENV = "MK_SPACY_GPU"

_lock = threading.Lock()
#: Every parse of every pipeline ``_load`` hands out holds this one lock. Its holders: ``nlp(text)`` and
#: ``nlp.pipe(...)`` (the class ``_load`` gives the pipeline) and ``g1_render._parse_tokens`` (a doc built from words
#: and run through the components); the parser service loads its own pipeline and does not take it. Why: a
#: transformer pipeline run from two threads at once can parse differently from a serial run (borderline attributes),
#: and serve rebuilds stored keys at open in worker threads (a drifted key would read as a stale index). Parses in
#: separate processes (capture shards) are untouched. A leaf: nothing inside a parse takes another kernel lock (no
#: kernel spaCy component), so no caller holding the WordNet, shard ledger or encoder lock can deadlock on it.
#: Reentrant, should a component ever parse.
PARSE_LOCK = threading.RLock()
#: ``model name -> its one loaded pipeline``. Every purpose, every module, reads the same copy.
_models: dict[str, object] = {}
#: ``(purpose, configured model) -> the pipeline that purpose reads on`` (the fallback decision is made once).
_resolved: dict[tuple[str, str], object] = {}
_fallback_noted: set[str] = set()
_gpu_asked = False


def pipeline(model: str):
    """The one loaded pipeline for ``model``, NER off, loaded at most once per process (thread-safe: the first load
    holds ``_lock``, so two threads never each load a copy). A load failure raises and is not
    remembered. NER is off because nothing consumes entity output (the attribute is read, e.g. ``ent_type_``, but is always empty; the parser
    service has its own loader and serves entities). Every spaCy reader in the kernel shares this copy (``nlp``, ``grammar.dependencies.load_spacy_model``,
    the event stages), so a process holds one transformer, not three."""
    found = _models.get(model)
    if found is None:
        with _lock:
            found = _models.get(model)
            if found is None:
                found = _models[model] = _load(model)
    return found


def _load(name: str):
    import spacy

    _ask_for_gpu(spacy)
    nlp = spacy.load(name, disable=["ner"])
    base = type(nlp)

    def call(self, text, *args, **kwargs):
        # trf raises ValueError ("Shape mismatch for blis.gemm") on empty or whitespace-only text: tokenize, don't run
        if isinstance(text, str) and not text.strip():
            return self.make_doc(text)
        with PARSE_LOCK:
            return base.__call__(self, text, *args, **kwargs)

    def pipe(self, texts, *args, **kwargs):
        # parsed whole under the lock, then handed out: a lock held across the generator's yields would stall every
        # other parse while the caller works on a doc
        with PARSE_LOCK:
            docs = list(base.pipe(self, texts, *args, **kwargs))
        return iter(docs)

    nlp.__class__ = type(base.__name__, (base,), {"__call__": call, "pipe": pipe})
    return nlp


def _ask_for_gpu(spacy) -> None:
    global _gpu_asked  # noqa: PLW0603
    if _gpu_asked or os.getenv(GPU_ENV, "").strip() != "1":
        return
    got = spacy.prefer_gpu()
    _gpu_asked = True  # only once prefer_gpu() has returned: a raise leaves the next load to ask again
    logger.info("%s=1: spaCy prefer_gpu() -> %s", GPU_ENV, got)


def nlp(purpose: Purpose = "extraction"):
    """The shared spaCy pipeline for ``purpose`` (see ``Purpose``). The model is ``spacy_model(purpose)``; the policy is
    what happens when it will not load (not installed (E050), a missing package, a missing factory (E002), any other error):
    ``extraction`` raises; ``turn`` and ``query`` log a warning and read on ``en_core_web_sm`` instead (``parser_note``
    says so), and raise only when the small model is the one that is missing."""
    from memory_kernel.grammar.dependencies import SPACY_FALLBACK_MODEL, spacy_model

    name = spacy_model(purpose)
    resolved = _resolved.get((purpose, name))
    if resolved is None:
        try:
            resolved = pipeline(name)
        except Exception as exc:  # noqa: BLE001  # whatever made the configured model fail: only extraction fails loud
            if purpose == "extraction" or name == SPACY_FALLBACK_MODEL:
                raise
            logger.warning(
                "spaCy model %s failed to load (%s: %s); %s parsing falls back to %s",
                name,
                type(exc).__name__,
                exc,
                purpose,
                SPACY_FALLBACK_MODEL,
            )
            resolved = pipeline(SPACY_FALLBACK_MODEL)
            _fallback_noted.add(purpose)
        _resolved[(purpose, name)] = resolved
    return resolved


def query_doc(text: str):
    """``nlp("query")(text)`` for a serve-time reader: never raises over a model. ``None`` only when no model at all can
    load (the caller reads that as "nothing to read")."""
    try:
        return nlp("query")(text)
    except Exception as exc:  # noqa: BLE001  # never raises over a model, whatever its failure
        logger.warning("no spaCy model can load (%s: %s); query-time reading is skipped", type(exc).__name__, exc)
        return None


#: The subject a bare command is read behind: "hum a tune" is parsed as "you hum a tune" (see ``turn_doc``). An
#: imperative's understood subject is the addressee; the frame token is dropped and is never the actor.
TURN_FRAME = "you "


def turn_doc(text: str):
    """``nlp("turn")(text)``, with the text stripped and a bare command read behind a subject.

    The parse is of ``text.strip()`` and the Doc returned is over the ORIGINAL ``text`` (leading and trailing whitespace
    are space tokens, every other token keeps its attributes and its offset shifted by the leading whitespace), so
    callers see the offsets of the text as written and " hum a tune" reads as "hum a tune" does.

    A tagger's first word of a bare imperative is often not a verb ("hum a tune" opens as an interjection). When the
    raw parse has no subject and does not open on a verb, the text is parsed again as ``"you " + text``; that parse is
    used only if the token after the frame is a VERB, and it is rebuilt without the frame token."""
    parser = nlp("turn")
    core = text.strip()
    if not core:
        return parser.make_doc(text)
    doc = _framed(parser, core)
    return doc if core == text else _widened(doc, text)


def _framed(parser, core: str):
    doc = parser(core)
    if len(doc) == 0 or doc[0].pos_ == "VERB" or any(t.dep_.startswith("nsubj") for t in doc):
        return doc
    framed = parser(TURN_FRAME + core)
    if len(framed) < 2 or framed[0].text.lower() != TURN_FRAME.strip() or framed[1].pos_ != "VERB":
        return doc
    tokens = framed[1:]
    root = next((t.i - 1 for t in tokens if t.head.i == t.i), 0)
    heads = [root if t.head.i == 0 else t.head.i - 1 for t in tokens]
    # the framed verb is an imperative, not the present tense the frame made it: base form, no agreement features
    tags = [t.tag_ for t in tokens]
    morphs = [str(t.morph) for t in tokens]
    tags[0] = "VB"
    morphs[0] = "VerbForm=Inf"
    return _rebuilt(
        doc,
        words=[t.text for t in tokens],
        spaces=[bool(t.whitespace_) for t in tokens],
        pos=[t.pos_ for t in tokens],
        tags=tags,
        morphs=morphs,
        lemmas=[t.lemma_ for t in tokens],
        heads=heads,
        deps=[t.dep_ for t in tokens],
    )


def _rebuilt(like, **attrs):
    from spacy.tokens import Doc

    out = Doc(like.vocab, **attrs)
    return out if out.text == like.text else like


def _widened(doc, text: str):
    """``doc`` (a parse of ``text.strip()``) as a Doc over ``text``: the whitespace around it becomes space tokens
    attached to the first and last word (as spaCy's own parse attaches them), nothing else moves. If the rebuilt text
    would differ from ``text`` the text is parsed as written."""
    from spacy.tokens import Doc

    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    last_space = trail.startswith(" ")  # as spaCy tokenizes: one space stays the last word's, the rest is a token
    trail = trail[1:] if last_space else trail
    shift = 1 if lead else 0
    words = [t.text for t in doc]
    spaces = [bool(t.whitespace_) or (last_space and t.i == len(doc) - 1) for t in doc]
    attrs = {
        "pos": [t.pos_ for t in doc],
        "tags": [t.tag_ for t in doc],
        "morphs": [str(t.morph) for t in doc],
        "lemmas": [t.lemma_ for t in doc],
        "heads": [t.head.i + shift for t in doc],
        "deps": [t.dep_ for t in doc],
    }
    if lead:  # attached to the first word
        words.insert(0, lead)
        spaces.insert(0, False)
        for key, value in zip(attrs, ("SPACE", "_SP", "", lead, shift, "dep"), strict=True):
            attrs[key].insert(0, value)
    if trail:  # attached to the last word
        last = len(words) - 1
        words.append(trail)
        spaces.append(False)
        for key, value in zip(attrs, ("SPACE", "_SP", "", trail, last, "dep"), strict=True):
            attrs[key].append(value)
    out = Doc(doc.vocab, words=words, spaces=spaces, **attrs)
    return out if out.text == text else nlp("turn")(text)


def parser_note(purpose: Purpose) -> str | None:
    """``"sm-fallback"`` once ``purpose`` had to fall back to the small model, else None (for the TurnPlan trace)."""
    return PARSER_FALLBACK if purpose in _fallback_noted else None


def reset_pipelines() -> None:
    """Forget the loaded pipelines, the fallback decisions and the notes (a test, or a changed model setting)."""
    with _lock:
        _models.clear()
        _resolved.clear()
        _fallback_noted.clear()


def normalize_exact(name: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"[\s]+", " ", name.lower()).strip()


@lru_cache(maxsize=4096)
def lemma_key(surface: str) -> str:
    """Exact-normalized lemma phrase of a stripped surface, parsed alone and lowercased.

    spaCy lemmas depend on context and casing ("with city planners" vs "city planners", "Our neighbours" vs "our
    neighbours"), so the key is always taken from the same canonical input: the stripped surface in lowercase.
    """
    doc = nlp()(normalize_exact(surface))
    return normalize_exact(" ".join(t.lemma_ for t in doc if not t.is_punct)) or normalize_exact(surface)


def identifying(det) -> bool:
    """A determiner that picks out one identifiable referent: definite, demonstrative or possessive (spaCy morph)."""
    morph = det.morph
    return "Def" in morph.get("Definite", []) or "Dem" in morph.get("PronType", []) or "Yes" in morph.get("Poss", [])


def non_identifying(span) -> bool:
    """A common-noun phrase with no identifying content: concern (4) of the remap ("lot", "people", "others").

    True when the head is a NOUN (never PROPN, never an all-caps initialism) and either the head itself is an
    indefinite pronominal (``PronType=Ind``), or it has no dependent except determiners, none of which is
    identifying, and it is either determined ("a lot", "a person", "some people": an indefinite or quantified
    singular introduces a new, non-identifiable referent) or a bare plural ("people", "others", "Lots": a generic
    kind reading). A bare singular ("music", "chess") is kept as the name of a kind or activity; any modifier,
    compound, possessor or prepositional dependent ("a chess club", "my family") makes it identifying.
    """
    head = span.root
    if head.pos_ != "NOUN" or (head.is_upper and len(head.text) > 1):
        return False
    if "Ind" in head.morph.get("PronType", []):
        return True
    deps = [c for c in head.children if c.i >= span.start and c.dep_ != "punct"]
    if any(c.dep_ != "det" for c in deps) or any(identifying(c) for c in deps):
        return False
    return bool(deps) or "Plur" in head.morph.get("Number", [])
