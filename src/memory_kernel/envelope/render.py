"""Envelope-driven statement rendering: first-person utterance -> third-person, deixis resolved.

The served corpus for a bonfire is a list of *statements*.  A statement minted from a
chat turn is first-person and deictic ("I went there last week"), which is unusable as a
retrieval surface: the speaker is not in the text, and "last week" is not a date.  The
envelope (speaker, addressee, timestamp) carries exactly the information needed to
resolve those indexicals, so we render the statement against the envelope instead of
storing the raw utterance.

`render()` is deterministic: same (utterance, speaker, addressee, ts) -> same string, no
I/O beyond loading the kernel's spaCy model and nltk WordNet once (lazily, on the first text
that needs them), no arguments mutated.  It is a *rule-based* span rewriter whose verb
agreement and preposition decisions read a dependency parse instead of word lists (the kernel keeps
no hand-written vocabulary lists).

------------------------------------------------------------------------------------
RULE TABLE
------------------------------------------------------------------------------------

S = speaker name (title-cased), A = addressee name (title-cased), P = "S and A".
D  = date of ts (YYYY-MM-DD).  MON(w) = Monday of week w.  All week math is
Monday-anchored (ISO).  A replacement inherits the capitalisation of the first
character of the span it replaced.

--- Person deixis: first person -------------------------------------------------
| source          | rendered            | note                                    |
| I               | S                   |                                         |
| I <verb>        | S <verb-3sg>        | the clause's FINITE verb (VBP/VBD) whose |
|                 |                     | nsubj is I, see VERB AGREEMENT below;    |
|                 |                     | words between I and the verb are copied  |
|                 |                     | ("I really want" -> "S really wants")    |
| I am / I'm      | S is                | 'm is unambiguously a contraction of am, |
|                 |                     | so the copula is expanded and agreed.    |
| I have / I've   | S has               |                                         |
| I'll            | S will              |                                         |
| I'd             | S had               | when the next verb of its chain is a     |
|                 |                     | past participle (tag VBN), or spaCy reads|
|                 |                     | 'd itself as past (VBD)                  |
| I'd             | S would             | otherwise ("I'd love", "I'd have loved") |
| I don't         | S doesn't           | likewise haven't -> hasn't, do -> does   |
| I was / I went  | S was / S went      | past tense: only be has a 3sg past form  |
| I can/could/... | S can/could/...     | modals (MD) are never inflected          |
| me, myself      | S                   | reflexivity is lost (documented loss).   |
|                 |                     | "me" inside a hyphenated compound        |
|                 |                     | ("me-time") is NOT a pronoun and is kept |
| my              | S's                 | possessive; name ending in s -> "s'"     |
| mine            | S's                 |                                          |

--- Person deixis: second person (only when there is exactly ONE addressee) ------
| you             | A                   | left verbatim when addressee is None     |
| you <verb>      | A <verb-3sg>        | same structural rule; a bare infinitive  |
|                 |                     | is not finite (VerbForm=Inf), so "makes  |
|                 |                     | you feel" -> "makes A feel", never       |
|                 |                     | "makes A feels", with no verb list       |
| you are / you're| A is                |                                          |
| you were        | A was               |                                          |
| you have/you've | A has               |                                          |
| you'll / you'd  | A will / A would    |                                          |
| you both/two/   | P                   | plural address = both participants       |
|   all/guys      |                     |                                          |
| your            | A's                 |                                          |
| yours, yourself | A's, A              |                                          |

--- Person deixis: first person plural (only when there is an addressee) ---------
| we              | P                   | verb stays plural (P is a plural NP)     |
| we/us all|both| | P                   | the quantifier is absorbed: "we all have"|
|   two|guys      |                     | -> "S and A have", not "S and A all have"|
| we're/we've/    | P are / P have /    |                                          |
|   we'll/we'd    | P will / P would    |                                          |
| us, ourselves   | P                   |                                          |
| our, ours       | P's                 | "our community" -> "S and A's community" |
|                 |                     | (over-specific when "our" is collective; |
|                 |                     | see the report's unresolved section)     |

--- Temporal deixis (resolved from ts) -------------------------------------------
| today                        | on D                                            |
| tonight                      | on the evening of D                             |
| this morning/afternoon/      | on the morning/afternoon/evening of D           |
|   evening                    |                                                 |
| yesterday                    | on D-1                                          |
| last night                   | on the night of D-1                             |
| tomorrow                     | on D+1                                          |
| tomorrow morning/night       | on the morning/night of D+1                     |
| this week                    | the week of MON(this week)                      |
| last week                    | the week of MON(this week)-7                    |
| next week                    | the week of MON(this week)+7                    |
| this weekend                 | the weekend of MON(this week)+5   (Saturday)    |
| last weekend / this past     | the weekend of MON(this week)-2                 |
|   weekend                    |                                                 |
| next weekend                 | the weekend of MON(this week)+12                |
| this month / last / next     | in YYYY-MM (of that month; the preposition keeps |
|                              | "a concert next month" readable)                |
| this year / last / next      | in YYYY                                         |
| last <weekday>               | on <date of the most recent earlier weekday>    |
| N days ago  (digit or one    | on D-N                                          |
|   number word of temporal_deixis.NUMBER_WORD_VALUES; "a few" is NOT resolved)    |
| in N days                    | on D+N                                          |
| N weeks ago / in N weeks     | the week of MON(D-7N) / MON(D+7N)               |

The leading "on "/"in " of a resolved date is dropped when the span already sits
behind a preposition -- the token before it is an ADP (or parsed as `prep`, e.g. SCONJ
"since") that governs the span; a particle, "went out last night", is not one: "this photo is from last night" -> "... from the night of
D-1", never "... from on the night of D-1".  Weekday names come from `calendar.day_name`.

--- VERB AGREEMENT (structural; no verb list) -------------------------------------
For a bare I / you that the parse makes the nsubj/nsubjpass of a clause, with no
coordinated partner ("you and I are" stays plural): the finite verb of that clause is its
first finite aux/auxpass, else the head itself when finite; a modal (MD) is never touched.
VBP -> its 3sg present; VBD -> its 3sg past only where one differs (were -> was).  The 3sg
form is the WordNet `verb.exc` form of the lemma that spaCy itself analyses as 3sg ("It
has", "It is", "It was"), else regular orthography (-ies after consonant+y; -es after
s/x/z/ch/sh/o; -s otherwise).  Only lemmas WordNet knows as verb base forms are inflected,
so tagger noise ("wanna", "kinda" tagged VBP) is copied verbatim.  The verb may precede
the pronoun ("Where have you been" -> "Where has A been", "Do you" -> "Does A").  The parse
runs once over the whole (tag-stripped) text, so offsets agree across quoted spans; a
verb inside a quoted span is never rewritten.

--- Deliberately NOT resolved (left verbatim, counted in the report) -------------
recently, lately, the other day, a few days ago, a while back, soon, these days,
someday, the past few weeks, third-person pronouns (he/she/they/it), "here"/"there".
Anything not listed above is copied through byte-for-byte.

--- Protected spans ---------------------------------------------------------------
Text inside double quotes ("..." and typographic "..."), and inside single quotes
when the quotes look like a quotation rather than an apostrophe (opening quote at a
word start preceded by start/space/(/, and closing quote at a word end followed by
space or punctuation, with no apostrophe inside), is copied verbatim.  In this corpus
those spans are titles ("The Long Road", 'Quiet Street Stories') where a
first-person pronoun belongs to the title, not to the speaker.

--- Invariants -------------------------------------------------------------------
* Nothing is ever deleted: every rule is an in-place span rewrite.
* One left-to-right pass; replacements are never rescanned, so a name substituted for
  "I" can never be re-matched by a later rule.
* speaker is required; addressee may be None, in which case every second-person and
  first-person-plural rule is a no-op and the text keeps "you"/"we"/"our".

--- Provenance and single authority -----------------------------------------------
Originally ported VERBATIM from the offline renderer script that produced the first
development-conversation artifact. This
module is the ONE envelope renderer: no other module under
`src/memory_kernel` defines an envelope `render(`.

**It is never called on the serve path.** The render is produced once, upstream, into
the corpus row's `utterance_rendered` field, which the search recipe only READS. Rendering at serve
time would double-render rows the corpus already holds rendered.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from collections.abc import Callable
from contextvars import ContextVar
from functools import cache
from typing import TYPE_CHECKING, Any

from memory_kernel.grammar import lexicon
from memory_kernel.grammar.semantic_operators import future_auxiliaries
from memory_kernel.graph.g1_nlp import Purpose

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

__all__ = ["render", "resolve_temporal"]

# ---------------------------------------------------------------------------
# temporal.  (Person deixis is read from the parse, not from patterns: see `_person_edits`.)  `{weekday}` and `{number}` are filled from resources (stdlib
# `calendar.day_name`, `knowledge.temporal_deixis.NUMBER_WORD_VALUES`) when the
# master pattern is first compiled -- see `_master()`.
# ---------------------------------------------------------------------------

_TEMPORAL_SRC = r"""
(?P<t_this_part>\bthis\s+(?:{this_part})\b)
|(?P<t_tomorrow_part>\btomorrow\s+(?:{tomorrow_part})\b)
|(?P<t_last_night>\blast\s+night\b)
|(?P<t_tonight>\btonight\b)
|(?P<t_today>\btoday\b)
|(?P<t_yesterday>\byesterday\b)
|(?P<t_tomorrow>\btomorrow\b)
|(?P<t_past_weekend>\bthis\s+past\s+weekend\b)
|(?P<t_weekend>\b(?P<t_weekend_det>{det})\s+weekend\b)
|(?P<t_week>\b(?P<t_week_det>{det})\s+week\b)
|(?P<t_month>\b(?P<t_month_det>{det})\s+month\b)
|(?P<t_year>\b(?P<t_year_det>{det})\s+year\b)
|(?P<t_last_wd>\blast\s+(?P<t_last_wd_name>{weekday})\b)
|(?P<t_days_ago>\b(?P<t_days_ago_n>\d+|{number})\s+days?\s+ago\b)
|(?P<t_in_days>\bin\s+(?P<t_in_days_n>\d+|{number})\s+days?\b)
|(?P<t_weeks_ago>\b(?P<t_weeks_ago_n>\d+|{number})\s+weeks?\s+ago\b)
|(?P<t_in_weeks>\bin\s+(?P<t_in_weeks_n>\d+|{number})\s+weeks?\b)
"""


def _weekday_names() -> list[str]:
    """Monday-first lowercase weekday names (stdlib `calendar`, index == `date.weekday()`)."""
    return [name.lower() for name in calendar.day_name]


@cache
def _master() -> re.Pattern[str]:
    from memory_kernel.knowledge.temporal_deixis import NUMBER_WORD_VALUES

    number = "|".join(sorted((re.escape(w) for w in NUMBER_WORD_VALUES), key=len, reverse=True))
    weekday = "|".join(re.escape(w) for w in _weekday_names())
    temporal = (
        _TEMPORAL_SRC.replace("{weekday}", weekday)
        .replace("{number}", number)
        .replace("{det}", "|".join(lexicon.relative_determiners()))
        .replace("{this_part}", "|".join(lexicon.dayparts_after_this()))
        .replace("{tomorrow_part}", "|".join(lexicon.dayparts_after_tomorrow()))
    )
    return re.compile(temporal, re.IGNORECASE | re.VERBOSE)


# quoted spans (protected)
_QUOTED = re.compile(
    r'"[^"]*"'
    r"|“[^”]*”"
    r"|(?:(?<=^)|(?<=[\s(,—-]))'[^'\n]{2,60}'(?=[\s,.!?;:)]|$)"
)


def _norm_name(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    return name if name[:1].isupper() else name[:1].upper() + name[1:]


def _poss(name: str) -> str:
    return name + ("'" if name.endswith(("s", "S")) else "'s")


def _match_case(src: str, repl: str) -> str:
    """Give `repl` the leading capitalisation of `src`."""
    if src[:1].isupper() and repl[:1].islower():
        return repl[:1].upper() + repl[1:]
    return repl


def _parse_ts(ts: str) -> _dt.date:
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(s).date()
    except ValueError:
        return _dt.date.fromisoformat(s[:10])


def _monday(d: _dt.date) -> _dt.date:
    return d - _dt.timedelta(days=d.weekday())


def _n(word: str) -> int:
    from memory_kernel.knowledge.temporal_deixis import relative_number_value

    value = relative_number_value(word)
    if value is None:  # the pattern only admits digits and NUMBER_WORD_VALUES keys
        raise ValueError(f"not a number: {word!r}")
    return value


def resolve_temporal(m: re.Match[str], day: _dt.date) -> str | None:
    """Resolve one temporal match against the envelope date, or None if not temporal."""
    g = m.group
    iso = day.isoformat()
    mon = _monday(day)

    if g("t_this_part"):
        part = m.group(0).split()[-1].lower()
        return f"on the {part} of {iso}"
    if g("t_tomorrow_part"):
        part = m.group(0).split()[-1].lower()
        return f"on the {part} of {(day + _dt.timedelta(days=1)).isoformat()}"
    if g("t_last_night"):
        return f"on the night of {(day - _dt.timedelta(days=1)).isoformat()}"
    if g("t_tonight"):
        return f"on the evening of {iso}"
    if g("t_today"):
        return f"on {iso}"
    if g("t_yesterday"):
        return f"on {(day - _dt.timedelta(days=1)).isoformat()}"
    if g("t_tomorrow"):
        return f"on {(day + _dt.timedelta(days=1)).isoformat()}"
    if g("t_past_weekend"):
        return f"the weekend of {(mon - _dt.timedelta(days=2)).isoformat()}"
    if g("t_weekend"):
        det = g("t_weekend_det").lower()
        offset = {"this": 5, "last": -2, "next": 12}[det]
        return f"the weekend of {(mon + _dt.timedelta(days=offset)).isoformat()}"
    if g("t_week"):
        det = g("t_week_det").lower()
        offset = {"this": 0, "last": -7, "next": 7}[det]
        return f"the week of {(mon + _dt.timedelta(days=offset)).isoformat()}"
    if g("t_month"):
        det = g("t_month_det").lower()
        if det == "this":
            target = day
        elif det == "last":
            target = day.replace(day=1) - _dt.timedelta(days=1)
        else:
            target = (day.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
        return f"in {target.year:04d}-{target.month:02d}"
    if g("t_year"):
        det = g("t_year_det").lower()
        return f"in {day.year + {'this': 0, 'last': -1, 'next': 1}[det]}"
    if g("t_last_wd"):
        target_wd = _weekday_names().index(g("t_last_wd_name").lower())
        back = (day.weekday() - target_wd) % 7 or 7
        return f"on {(day - _dt.timedelta(days=back)).isoformat()}"
    if g("t_days_ago"):
        return f"on {(day - _dt.timedelta(days=_n(g('t_days_ago_n')))).isoformat()}"
    if g("t_in_days"):
        return f"on {(day + _dt.timedelta(days=_n(g('t_in_days_n')))).isoformat()}"
    if g("t_weeks_ago"):
        return f"the week of {(mon - _dt.timedelta(days=7 * _n(g('t_weeks_ago_n')))).isoformat()}"
    if g("t_in_weeks"):
        return f"the week of {(mon + _dt.timedelta(days=7 * _n(g('t_in_weeks_n')))).isoformat()}"
    return None


# ---------------------------------------------------------------------------
# structure: the kernel's spaCy parse + WordNet's verb exceptions
# ---------------------------------------------------------------------------


#: Why ``render`` is parsing (``Purpose``): the build renders on ``extraction`` (fail loud), serve-time index keys on
#: ``query`` (a model that will not load degrades to the small one). Set by ``render`` for the length of one call.
_PURPOSE: ContextVar[Purpose] = ContextVar("envelope_render_purpose", default="extraction")


def _nlp(purpose: Purpose | None = None) -> Language:
    from memory_kernel.graph.g1_nlp import nlp

    return nlp(purpose or _PURPOSE.get())


def _wordnet() -> Any:  # pyright: ignore[reportExplicitAny, reportAny]  # nltk is untyped
    """The kernel's shared WordNet reader; LOUD when the corpus is missing.

    Verb agreement needs it for every bare I/you, so a host without the corpus must fail
    at the first render rather than silently emit unagreed text (an env-parity trap).
    """
    from memory_kernel.grammar.dependencies import wordnet_module

    wn = wordnet_module()  # pyright: ignore[reportAny]
    if wn is None:
        raise RuntimeError("envelope render needs the nltk WordNet corpus (verb agreement); it is not installed")
    return wn  # pyright: ignore[reportAny]


@cache
def _verb_exceptions() -> dict[str, tuple[str, ...]]:
    """WordNet `verb.exc` reversed: lemma -> its irregular inflected forms."""
    wn = _wordnet()  # pyright: ignore[reportAny]  # nltk is untyped
    exc: str = wn.open("verb.exc").read()  # pyright: ignore[reportAny]
    out: dict[str, list[str]] = {}
    for line in exc.splitlines():
        parts = line.split()
        for lemma in parts[1:]:
            out.setdefault(lemma, []).append(parts[0])
    return {lemma: tuple(forms) for lemma, forms in out.items()}


@cache
def _is_verb_base(lemma: str) -> bool:
    """WordNet knows `lemma` as the base form of a verb."""
    wn = _wordnet()  # pyright: ignore[reportAny]  # nltk is untyped
    base: str | None = wn.morphy(lemma, wn.VERB)  # pyright: ignore[reportAny]
    return lemma.isalpha() and base == lemma


@cache
def _irregular_3sg(lemma: str, tense: str, purpose: Purpose = "extraction") -> str | None:
    """The WordNet exception form of `lemma` that spaCy analyses as 3sg in `tense`, if any."""
    for form in _verb_exceptions().get(lemma, ()):
        tok = _nlp(purpose)(f"It {form}")[1]
        morph = tok.morph
        third = tense in morph.get("Tense", []) and "3" in morph.get("Person", [])
        if tok.lower_ == form and third and ("Sing" in morph.get("Number", []) or tok.tag_ == "VBZ"):
            return form
    return None


@cache
def _base_reads_present(lemma: str, purpose: Purpose = "extraction") -> bool:
    """The bare form is not also a past tense: spaCy does not read "It <lemma>" as VBD.

    Verbs whose base form doubles as their past ("put", "read", "bet", "cut") read as VBD
    after a 3sg subject (which a bare present cannot follow), so "This poem I read last
    year" never becomes "reads" -- the safety property the old closed verb table encoded
    by leaving such verbs out.
    """
    return bool(_nlp(purpose)(f"It {lemma}")[1].tag_ != "VBD")


def _regular_3sg(lemma: str) -> str:
    """Regular English orthography for the 3sg present."""
    if len(lemma) > 1 and lemma.endswith("y") and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    if re.search(r"(?:[sxzo]|[cs]h)$", lemma):  # sibilant or -o: -es
        return lemma + "es"
    return lemma + "s"


def _third_singular(verb: Token, core: str) -> str | None:
    """The 3sg form a finite verb takes when its I/you subject becomes a name, or None.

    `core` is the token text without trailing punctuation the tokenizer left fused on
    ("do-" in "yes, I do- Ben"); the lemma is trimmed the same way.
    """
    lemma = re.sub(r"[\W_]+$", "", verb.lemma_.lower())
    if not _is_verb_base(lemma):
        return None
    if verb.tag_ == "VBP":
        if not _base_reads_present(lemma, _PURPOSE.get()):
            return None
        form = _irregular_3sg(lemma, "Pres", _PURPOSE.get()) or _regular_3sg(lemma)
    elif verb.tag_ == "VBD":
        form = _irregular_3sg(lemma, "Past", _PURPOSE.get())
    else:
        return None
    if form is None or form == core.lower():
        return None
    if core.isupper() and len(core) > 1:
        return form.upper()
    return _match_case(core, form)


def _finite_verb_of(subject: Token) -> Token | None:
    """The finite verb that agrees with `subject` (nsubj of its clause), or None."""
    if subject.dep_ not in ("nsubj", "nsubjpass"):
        return None
    if any(child.dep_ in ("conj", "cc") for child in subject.children):
        return None  # "you and I are": a plural subject
    head = subject.head
    chain = sorted(
        [c for c in head.children if c.dep_ in ("aux", "auxpass") and "Fin" in c.morph.get("VerbForm", [])],
        key=lambda t: t.i,
    )
    if chain:
        verb = chain[0]
    elif "Fin" in head.morph.get("VerbForm", []) and head.pos_ in ("VERB", "AUX"):
        verb = head
    else:
        return None
    return None if verb.tag_ == "MD" else verb


def _d_is_had(clitic: Token) -> bool:
    """'d reads as "had" when the next verb of its chain is a past participle (or spaCy reads it as past)."""
    if clitic.tag_ == "VBD":
        return True
    head = clitic.head
    chain = sorted(
        [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i > clitic.i] + [head],
        key=lambda t: t.i,
    )
    after = [t for t in chain if t.i > clitic.i]
    return bool(after) and after[0].tag_ == "VBN"


class _Parse:
    """The whole-text parse, built on first use and indexed by character offset."""

    _text: str

    def __init__(self, text: str) -> None:
        self._text = text
        self._purpose = _PURPOSE.get()
        self._starts: dict[int, Token] = {}
        self._doc: Doc | None = None

    @property
    def text(self) -> str:
        return self._text

    def _ensure(self) -> Doc:
        if self._doc is None:
            self._doc = _nlp(self._purpose)(
                self._text.replace("’", "'")
            )  # same length: offsets hold, clitics get their lemma
            self._starts = {t.idx: t for t in self._doc}
        return self._doc

    def at(self, char: int) -> Token | None:
        """The token starting exactly at `char`."""
        _ = self._ensure()
        return self._starts.get(char)

    def before(self, char: int) -> Token | None:
        """The last non-space token ending at or before `char`."""
        prev = None
        for t in self._ensure():
            if t.idx + len(t.text) > char:
                break
            if not t.is_space:
                prev = t
        return prev


def _behind_preposition(parse: _Parse, start: int, end: int) -> bool:
    """The span [start, end) is the object of the preposition right before it.

    A preposition is an ADP, or any token the parser attaches as `prep` ("since" is SCONJ
    by POS but heads a PP); a verb particle (`prt`: "went out last night") is not one.
    """
    prev = parse.before(start)
    if prev is None or prev.dep_ == "prt" or not (prev.pos_ == "ADP" or prev.dep_ == "prep"):
        return False
    inside = [t for t in prev.doc if start <= t.idx < end]
    return any(t.head.i == prev.i for t in inside)


def _adjacent(left: Token, right: Token | None) -> bool:
    return right is not None and left.idx + len(left.text) == right.idx


def _clitic_after(pronoun: Token) -> Token | None:
    """The auxiliary clitic the tokenizer split off the pronoun ("I" + "'m", "you" + "'ve"), or None.

    The clitic is the next token when it touches the pronoun, starts with an apostrophe and is an auxiliary or a verb.
    """
    nxt = pronoun.nbor() if pronoun.i + 1 < len(pronoun.doc) else None
    if nxt is None or not _adjacent(pronoun, nxt) or nxt.text[:1] != "'" or nxt.pos_ not in ("AUX", "VERB"):
        return None
    return nxt


def _is_d(clitic: Token) -> bool:
    """The clitic 'd (would, or had when the parse reads it as past): a modal (MD) that is not a future auxiliary
    ('ll is one: tense.json), or a past tense (VBD)."""
    return clitic.tag_ == "VBD" or (clitic.tag_ == "MD" and clitic.lemma_.lower() not in future_auxiliaries())


def _plural_marker(pronoun: Token) -> Token | None:
    """The token that makes "you both", "we all", "us two", "you guys" a plural address, or None.

    It follows the pronoun after whitespace and is either a determiner, numeral or "both" the parse attaches as an
    apposition, a numeral or a dependent (never the count of a plural noun that comes after it: "us two hours"), or the plural
    common noun the pronoun modifies ("guys").
    """
    if pronoun.whitespace_ == "" or pronoun.i + 1 >= len(pronoun.doc):
        return None
    nxt = pronoun.nbor()
    if nxt.tag_ == "NNS" and pronoun.head.i == nxt.i and pronoun.dep_ == "nmod":
        return nxt
    if nxt.tag_ in ("DT", "CD") or (nxt.tag_ == "CC" and nxt.dep_ == "preconj"):
        before_noun = nxt.head.i > nxt.i and nxt.head.tag_ in (
            "NNS",
            "NNPS",
        )  # "us two hours": a count of a plural noun
        if nxt.dep_ in ("appos", "nummod", "dep", "preconj") and not before_noun and _counts_two_or_more(nxt):
            return nxt
    return None


def _counts_two_or_more(marker: Token) -> bool:
    """A numeral marker names at least two people ("you two", not "you one day"); a determiner or "both" is not a numeral."""
    if marker.tag_ != "CD":
        return True
    from memory_kernel.knowledge.temporal_deixis import relative_number_value

    count = relative_number_value(marker.text)
    return count is not None and count >= 2


def _fused(token: Token) -> bool:
    """The tokenizer cut this pronoun out of a longer word ("Wed" -> "We" + "d"): the next token touches it and is a word."""
    return token.i + 1 < len(token.doc) and _adjacent(token, token.nbor()) and token.nbor().text[:1].isalnum()


def _hyphen_compound(token: Token) -> bool:
    """A pronoun form inside a hyphenated compound ("me-time") is a modifier, not a pronoun."""
    return token.i + 1 < len(token.doc) and _adjacent(token, token.nbor()) and token.nbor().tag_ == "HYPH"


def _person_edits(
    speaker: str,
    addressee: str | None,
    *,
    parse: _Parse,
    start: int,
    end: int,
    verb_edits: dict[int, tuple[int, str]],
) -> list[tuple[int, int, str]]:
    """The rewrites of the first- and second-person pronouns in text[start:end], read from the parse.

    A pronoun is a token the tagger calls PRP or PRP$; its Person, Number, Poss and Reflex come from its morphology (the
    pronoun table fills in a Person the parse left off). First person is the speaker (a pair, with plural number, when
    there is an addressee); second person is the addressee; both need the addressee except the singular first person.
    """
    from memory_kernel.grammar.pronouns import clitic_form, of_token

    both = f"{speaker} and {addressee}" if addressee else None
    doc = parse._ensure()
    tokens = [t for t in doc if start <= t.idx and t.idx + len(t.text) <= end]
    spans: list[tuple[int, int, str]] = []

    def agree(subject: Token) -> None:
        verb = _finite_verb_of(subject)
        if verb is None or not (start <= verb.idx and verb.idx + len(verb.text) <= end):
            return
        core = re.sub(r"[^\w'’]+$", "", verb.text)
        form = _third_singular(verb, core) if core else None
        if form is None:
            return
        if verb.text[:1] in "'’" and verb.idx > 0 and not parse.text[verb.idx - 1].isspace():
            # an enclitic on the word before ("How're you", "What've you"): spell it out
            form = " " + form.lower()
        verb_edits[verb.idx] = (verb.idx + len(core), form)

    def had_or_would(clitic: Token, name: str) -> str:
        return f"{name} had" if _d_is_had(clitic) else f"{name} would"

    for t in tokens:
        pronoun = of_token(t)
        if pronoun is None or pronoun.person not in (1, 2) or _hyphen_compound(t) or _fused(t):
            continue
        if pronoun.person == 2 and pronoun.plural:
            continue  # "yourselves": not addressed by the single-addressee rules
        pair = pronoun.person == 1 and pronoun.plural  # "we", "us", "our": the speaker and the addressee
        if (pronoun.person == 2 or pair) and addressee is None:
            continue  # second person and first-person plural need an addressee
        name = both if pair else speaker if pronoun.person == 1 else addressee
        assert name is not None
        stop = t.idx + len(t.text)
        bare = not (pronoun.possessive or pronoun.reflexive)
        clitic = _clitic_after(t) if bare else None
        marker = _plural_marker(t) if bare and clitic is None and (pair or pronoun.person == 2) else None
        if clitic is not None:
            stop = clitic.idx + len(clitic.text)
            form = clitic_form(clitic.lemma_, plural=pair)
            if _is_d(clitic):
                text = f"{name} would" if pair else had_or_would(clitic, name)
            elif form is None:
                continue
            else:
                text = f"{name} {form}"
        elif marker is not None:
            assert both is not None
            stop = marker.idx + len(marker.text)
            text = _match_case(t.text, both)
        elif pronoun.possessive:
            text = _match_case(t.text, _poss(name)) if pair else _poss(name)
        else:
            if bare and not pair and "Acc" not in t.morph.get("Case"):
                agree(t)  # I/you as a subject: the finite verb agrees with the name
            text = _match_case(t.text, name) if pair else name
        spans.append((t.idx, stop, text))
    return spans


def _make_replacer(day: _dt.date, *, parse: _Parse, base: int) -> Callable[[re.Match[str]], str]:
    def repl(m: re.Match[str]) -> str:
        src = m.group(0)
        temporal = resolve_temporal(m, day)
        if temporal is None:
            return src
        if temporal.startswith(("on ", "in ")) and _behind_preposition(parse, base + m.start(), base + m.end()):
            temporal = temporal[3:]
        return _match_case(src, temporal)

    return repl


def render(utterance: str, *, speaker: str, addressee: str | None, ts: str, purpose: Purpose = "extraction") -> str:
    """Render a first-person utterance as third-person, deixis-resolved text.

    Deterministic: no mutation of arguments; the only I/O is the one-time lazy load of the
    kernel's spaCy model and WordNet.  See the module docstring for the full rule table.

    Args:
        utterance: the raw statement text (may carry a "Name: " prefix, which is
            stripped and replaced by the rendering).
        speaker:   who produced the utterance.  Required.
        addressee: the single addressee, or None when there is not exactly one (all
            second-person and first-person-plural rules then become no-ops).
        ts:        ISO-8601 timestamp of the utterance, the origin for temporal deixis.
        purpose:   why the text is parsed (``g1_nlp.Purpose``): ``extraction`` (the default, the build's) fails loud when
                   the model will not load; ``query`` (serve-time index keys) falls back to the small model.
    """
    token = _PURPOSE.set(purpose)
    try:
        return _render(utterance, speaker=speaker, addressee=addressee, ts=ts)
    finally:
        _PURPOSE.reset(token)


def _render(utterance: str, *, speaker: str, addressee: str | None, ts: str) -> str:
    if not utterance:
        return utterance
    spk = _norm_name(speaker)
    if spk is None:
        raise ValueError("render() requires a speaker name")
    adr = _norm_name(addressee)
    if adr is not None and adr.lower() == spk.lower():
        adr = None
    day = _parse_ts(ts)

    text = utterance
    # a leading "Name: " speaker tag is envelope data, not content
    tag = re.match(rf"^\s*{re.escape(spk)}\s*:\s*", text, re.IGNORECASE)
    if tag:
        text = text[tag.end() :]

    parse = _Parse(text)  # parsed on first use only
    master = _master()

    def _sub(start: int, end: int) -> str:
        # the replacer inspects the parse by absolute offset, so it is bound to the exact
        # segment it rewrites; verb edits land after the pronoun pass, never rescanned
        segment = text[start:end]
        verb_edits: dict[int, tuple[int, str]] = {}
        spans: list[tuple[int, int, str]] = []
        replacer = _make_replacer(day, parse=parse, base=start)
        for m in master.finditer(segment):
            spans.append((start + m.start(), start + m.end(), replacer(m)))
        for ps, pe, form in _person_edits(spk, adr, parse=parse, start=start, end=end, verb_edits=verb_edits):
            if not any(s < pe and ps < e for s, e, _ in spans):
                spans.append((ps, pe, form))
        for vs, (ve, form) in verb_edits.items():
            if not any(s < ve and vs < e for s, e, _ in spans):
                spans.append((vs, ve, form))
        spans.sort()
        out: list[str] = []
        pos = start
        for s, e, repl in spans:
            out.append(text[pos:s])
            out.append(repl)
            pos = e
        out.append(text[pos:end])
        return "".join(out)

    out: list[str] = []
    pos = 0
    for q in _QUOTED.finditer(text):
        out.append(_sub(pos, q.start()))
        out.append(q.group(0))
        pos = q.end()
    out.append(_sub(pos, len(text)))
    return "".join(out)
