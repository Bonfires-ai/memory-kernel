"""The keyword glossary: the runtime grammar's closed operation set, rendered from the code.

Every op spec of ``VALUE_SPECS`` (executor), ``MEMORY_SPECS`` (ops_memory) and ``LEXICON_SPECS`` (lexicon) is a
KEYWORD. ``glossary()`` renders each one from its spec (name, inputs and outputs, kind, side effect, declared state
cells) and from ``RULES`` here (what it reads, its rules text, how it fails), which quotes the constants the code runs
on (``WINDOW_RULES``, ``WITHIN_RULES``, ``EITHER_RULES``, ``CORE_ROLES`` / ``NON_SUBJECT``, ``EVENTS_LIMIT``, ``KIND_LABELS``,
``DATE_KEYS``). A spec with no ``RULES`` entry, or an entry with no spec, raises: a keyword is never added without its
rules text. ``mk keywords`` prints the glossary; ``docs/keywords.md`` is its output, and a test fails when that file
is stale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from memory_kernel.graph.g1_names import KIND_LABELS
from memory_kernel.runtime.executor import DATE_KEYS, EITHER_RULES, VALUE_SPECS, WINDOW_RULES, WITHIN_RULES
from memory_kernel.runtime.lexicon import LEXICON_SPECS, SHARED
from memory_kernel.runtime.ops_memory import CORE_ROLES, EVENTS_LIMIT, MEMORY_SPECS, NON_SUBJECT
from memory_kernel.runtime.program import OpSpec
from memory_kernel.runtime.teach import INVOKE_DEPTH

#: the three keyword sets, in glossary order
KEYWORD_SETS: tuple[tuple[str, Mapping[str, OpSpec]], ...] = (
    ("Value keywords", VALUE_SPECS),
    ("Memory keywords", MEMORY_SPECS),
    ("Lexicon keywords", LEXICON_SPECS),
)

GRAPH = "the bonfire's graph (read-only Cypher, scoped by its group)"
NOTHING = "nothing"


@dataclass(frozen=True)
class Rules:
    #: what the keyword reads
    reads: str
    #: its rules text
    text: str
    #: how it fails beyond the common rule ("" when only the common rule applies)
    fails: str = ""


_ROLES = ", ".join(f"`{r}`" for r in CORE_ROLES)
_KINDS = ", ".join(KIND_LABELS)
_DATES = ", ".join(DATE_KEYS)

RULES: Mapping[str, Rules] = {
    # -- value ---------------------------------------------------------------------------------------------------------
    "name-span": Rules(
        "the turn's text",
        "The surface text between two offsets of the normalised turn (a slot's boundaries), as it was written (case "
        "and punctuation kept), with its leading determiners dropped unless a capitalised one starts a proper name "
        '("The Brask Group" keeps "The"). A dropped quantifying determiner ("every team") makes the reading a repair '
        "before the program runs: no keyword applies the quantifier.",
        "offsets that are not integers, or a span that covers no token",
    ),
    "lemma-span": Rules(
        "the normalised turn",
        "The lemma of the one verb the normaliser marked between two offsets (the verb slot's text).",
        "a span that is not exactly one marked verb",
    ),
    "within": Rules("the turn's time (`at`)", WITHIN_RULES),
    "const": Rules(NOTHING, "Its input, a constant of the construction's meaning, as a value."),
    "either": Rules(NOTHING, EITHER_RULES),
    "gather": Rules(
        NOTHING,
        "Its inputs, any number of values (none gives an empty list), as one list in argument order; an input that is "
        "itself a list stays one element (nothing is flattened). It runs once every input is bound. A taught phrase "
        "uses it to fill a parameter that takes several values with the results of its earlier steps that fit that "
        'parameter ("make X from them").',
    ),
    "deixis": Rules(
        "the turn's speaker or addressee",
        "The turn's `speaker` or `addressee` (the role is the constant `speaker` or `addressee`), as the platform "
        'gives it: "me" is the speaker, "you" the addressee.',
        "refused before the program runs when the role is neither, or the turn has no such person (an API call with "
        "no speaker never runs a program that says me)",
    ),
    "everyone": Rules(
        "the turn's participants",
        "The turn's participants, as a set (a taught body's or a live turn's procedure repeats once per member, "
        "unless every parameter the set fills, directly or through an invoked phrase's slot, takes several values "
        "(`many`): then it takes the set whole).",
        "refused before the program runs when the turn has no participants, or when one set fills both a `many` "
        "parameter and one that takes one value (say it in two clauses; teaching refuses it too)",
    ),
    "window": Rules("the chat's turn window", WINDOW_RULES),
    "definite": Rules(
        NOTHING, "Marks its referent's noun phrase as definite (the). No output; no effect at execution."
    ),
    "indefinite": Rules(
        NOTHING, "Marks its referent's noun phrase as indefinite (a, an). No output; no effect at execution."
    ),
    "plural": Rules(
        NOTHING,
        'Marks its referent (a `window` referent: "them") as plural. In a taught phrase\'s body, teaching reads it '
        "as the results of the phrase's earlier steps that fit the parameter it fills, gathered (`gather`) into a "
        "parameter that takes several values (into one that takes one value it is refused), and stores no marker. "
        "No output; no effect at execution.",
        "refused before the program runs in a turn: 'them' refers to the results of earlier steps of a taught "
        "phrase, and a turn has none (a repair)",
    ),
    "procedure": Rules(
        NOTHING,
        "Names a procedure (a taught body): the variable its `seq` steps and its `distribute` hang on.",
    ),
    "seq": Rules(
        NOTHING,
        "Orders a step (a tool step's result, or a nested procedure) at an integer index in a procedure. The order is "
        "a contract: the first unresolved step of a procedure blocks every later step of it. Never runs, never traced.",
        "refused before the program runs when it orders something that is not a step of a procedure, or a step waits "
        "on a later one",
    ),
    "distribute": Rules(
        NOTHING,
        "Repeats a procedure once per member of a set, the member bound in place of the set; the data ops that depend "
        "on the set or on its tool steps' results, and feed its tool steps, run once per member too (a `gather` of a "
        "member's results runs in that member's turn). A set no `distribute` scopes reaches its readers whole. Never "
        "runs, never traced.",
        "refused before the program runs over a set computed inside a distributed procedure",
    ),
    "invoke": Rules(
        "the construction store (the invoked construction's live program)",
        "Runs a taught construction's current program as one step of this procedure; refused when it would invoke "
        "itself; repairs when that construction is gone.",
        "refused at teach time when it would invoke itself; a repair before the program runs when the construction "
        f"is no longer live (`invokes <id>, which is no longer live`) or the references nest deeper than {INVOKE_DEPTH}",
    ),
    # -- memory --------------------------------------------------------------------------------------------------------
    "find-entity": Rules(
        GRAPH,
        "The entities (not events) whose name is the given name exactly, else the same name in any case; each as "
        "{uuid, name, kinds}. A name the turn window already resolved (a Referent, `window`'s output) is returned as "
        "it is, with no query: it is never looked up again by name, and no other value passes through.",
        "a result of nothing makes the turn a repair (the name names nobody in the graph): never an answer built from "
        "another entity",
    ),
    "events": Rules(
        GRAPH,
        f"The events of a frame, or of any frame of a set (`frames-of`'s output), at most {EVENTS_LIMIT}, ordered by "
        "`happened_at`, then uuid. A participant (an entity, a uuid, or one name's candidates) narrows them to the events it takes part "
        "in; a list of such participants narrows them to the events that have ALL of them. A role is one of the closed "
        f"constants {_ROLES}, or `{NON_SUBJECT}` (any core role but `{CORE_ROLES[0]}`: a participant who is not the "
        "subject), or absent (any role); with several participants, one role for all or a list parallel to them. A "
        "participant that names nobody reads no event.",
        f"a role outside {_ROLES} and `{NON_SUBJECT}`, or a role list not parallel to the participants",
    ),
    "role-filler": Rules(
        GRAPH,
        "The entities that fill a role of the given events, each once. The role is one role, a list of roles, or the "
        "(frame, role) pairs `wh-role` returns (a pair reads its role only on events of its frame). It never returns "
        "an entity of its `except` input: the participants the question gives.",
        "a role that is neither a string nor a (frame, role) pair",
    ),
    "identity-lines": Rules(
        GRAPH,
        "The lines of the entities' summaries that say who the entity is: copular statements with a nominal predicate "
        "whose subject is the entity (or the speaker's own first person).",
    ),
    "state-lines": Rules(
        GRAPH,
        "The entity's summary lines about a frame. Summary lines carry no frame yet, so today it returns the identity "
        "lines (`identity-lines`).",
    ),
    "slot-of-kind": Rules(
        GRAPH,
        f"The graph entities of one store kind ({_KINDS}), or of any kind of a set, that the lines name, whole words "
        "in any case, longest non-overlapping matches only. The kind is a constant of the construction's meaning or "
        "the set `role-kinds` returns, each matched to the store's label whatever its case; an empty set reads "
        "nothing (no query).",
        f"a kind that is not one of {_KINDS}",
    ),
    "time-of": Rules(
        GRAPH,
        "An event's date (its `happened_at`, else its `told_at`), or the list of them for a list of events; None for "
        "an undated one.",
    ),
    "source": Rules(
        GRAPH,
        "Where an item was said: per item, each message that mentions it (the item itself when it is a message) and "
        "the saga holding that message, with the saga's name.",
    ),
    "mentions": Rules(GRAPH, "The messages that mention the entity: uuid, content, sender and date, in date order."),
    "count": Rules(NOTHING, "The number of items in a set (no query)."),
    "order": Rules(
        NOTHING,
        f"The first or the latest item of a set by date (its first non-empty {_DATES}); undated items are dropped; "
        "None when none is dated. The end is the constant `first` or `latest` (no query).",
        "an end that is neither `first` nor `latest`",
    ),
    "recall": Rules(
        "the bonfire's `/context` retrieval",
        "The fallback: the context block today's retrieval gives for the text (first-person words resolved to the "
        "turn's speaker). The runtime runs it on a repair whose result was empty and puts it in the plan's "
        "`memory_results.recall`. In `mk serve` it reads the context caches only, never a paid call: a cache miss "
        "leaves no `recall` result and the diagnostic `recall unavailable: no cached context`.",
        "no context callable configured, or no cached context (the repair stands without it)",
    ),
    # -- lexicon -------------------------------------------------------------------------------------------------------
    "frames-of": Rules(
        "the lexicon: the bonfire's rolesets (`chain/rolesets.json`) over the shipped question map",
        "The rolesets of a verb lemma: the bonfire's own, else the question map's, sorted; a roleset with no role is "
        "never returned. Verb-sense ambiguity travels on as the whole set.",
        "a result of nothing makes the turn a repair (the verb names no frame): never a polar no",
    ),
    "wh-role": Rules(
        "the lexicon's question map",
        f"The (frame, role) pairs that answer a wh constant in the given frames, frame by frame: each frame's own roles, "
        f"then the modifiers every frame admits (the map's `{SHARED}` entry). A label joining two kinds (`who_or_what`) "
        "answers either; `unknown` answers none.",
    ),
    "role-kinds": Rules(
        "the lexicon's question map",
        f"The store kinds ({_KINDS}) of the entities that fill the given (frame, role) pairs (`wh-role`'s output), "
        "sorted: per frame, the kinds of its own roles among the pairs; a frame with none of its own roles among them "
        f"takes the kinds of the modifiers every frame admits (the map's `{SHARED}` entry). A role's kinds are generated "
        "from its VerbNet themes (`location`, `destination`, `source`, `initial_location` and `trajectory` are a "
        "Place; a `beneficiary` is an Organisation; the shared `argm-loc` is a `location`); a role with no such "
        "theme has none. An empty result means no kind is expected: a `slot-of-kind` over it reads nothing.",
        "a role that is not a (frame, role) pair",
    ),
}


def signature(spec: OpSpec) -> str:
    """``name(outputs ← inputs)``; with no output or no input, the arguments alone (``definite(referent)``,
    ``everyone(people)``: the entry's ``outputs`` / ``inputs`` line tells which)."""
    if not spec.outputs or not spec.inputs:
        return f"{spec.name}({', '.join([*spec.outputs, *spec.inputs])})"
    return f"{spec.name}({', '.join(spec.outputs)} ← {', '.join(spec.inputs)})"


def _side_effect(spec: OpSpec) -> str:
    return "none" if spec.side_effect is False else "unknown" if spec.side_effect is None else "yes"


def _check() -> None:
    names = [name for _, specs in KEYWORD_SETS for name in specs]
    missing = sorted(set(names) - set(RULES))
    extra = sorted(set(RULES) - set(names))
    if missing or extra or len(names) != len(set(names)):
        raise KeyError(f"keyword rules out of step with the specs: missing {missing}, no spec {extra}")


HEADER = """# Runtime grammar keywords

Generated from the code by `mk keywords` (`memory_kernel.runtime.keywords`); a test fails when this file is stale.
Do not edit it by hand: change the code and run `mk keywords > docs/keywords.md`.

The runtime grammar's operations are a closed set of **keywords**. A construction's meaning is a program that composes
keywords (and the tools of the catalogue the agent runtime declares); it never defines one. The set changes only in a
kernel release: teaching and remining compose keywords and never add one (the program validator refuses an unknown op).
This page is also the reference an LLM reads when it drafts a construction.

**Reading an entry.** `name(outputs ← inputs)`: in a meaning predicate the first arguments are the outputs, the rest
the inputs (`["find-entity", "?e", "?x"]` writes `?e` from `?x`). A keyword with no output or no input shows its
arguments alone, and its `outputs` / `inputs` line says which. An input is a variable another keyword writes, or a
constant of the meaning. Kinds: `value` (computed from the turn itself), `memory` (a read of the graph, the lexicon
or today's retrieval), `control` (the structure of a taught body's tool steps).

**No keyword writes or has a side effect.** The kernel only reads. The one effect a program can have is a tool step,
and **the kernel never runs a tool**: it returns the step in the plan for the agent runtime to run (a tool whose
`side_effect` is true or unknown is marked `needs_confirmation`), and the runtime reports the result with `resume`.

**Failure.** A keyword that fails ends the run: the turn becomes a `repair` with the diagnostic
`<keyword> (op N) failed: <reason>`, and nothing after it runs. A refusal "before the program runs" is a repair in
which no keyword ran. A repair is never a guessed answer: the agent runtime falls back to today's path.
"""

TOOLS = """## Tools

Tools are not keywords. The agent runtime declares its catalogue (`ToolSpec`: name, args, `side_effect` true, false
or unknown, and the state cells it `reads` and `writes`); a construction may then use a tool like a keyword whose one
output is the tool's result. The kernel stops at the first tool step whose inputs are bound and returns it in the
plan's `tool_steps`; the plan's `licence` lists every tool the program uses with its declared cells (surfaced, never
enforced: the agent runtime's resolver enforces them).
"""


def glossary() -> str:
    """The keyword glossary as Markdown (``docs/keywords.md``)."""
    _check()
    parts = [HEADER]
    for title, specs in KEYWORD_SETS:
        parts.append(f"## {title}\n")
        for name in sorted(specs):
            spec, rules = specs[name], RULES[name]
            writes = ", ".join(spec.writes) or NOTHING
            lines = [
                f"### `{signature(spec)}`",
                "",
                f"- outputs: {', '.join(spec.outputs) or 'none'}; inputs: {', '.join(spec.inputs) or 'none'}",
                f"- kind: `{spec.kind}`; side effect: {_side_effect(spec)}; writes: {writes}",
                f"- reads: {', '.join([rules.reads, *spec.reads])}",
            ]
            if rules.fails:
                lines.append(f"- fails: {rules.fails}")
            lines += ["", rules.text, ""]
            parts.append("\n".join(lines))
    parts.append(TOOLS)
    return "\n".join(parts)


__all__ = ["KEYWORD_SETS", "RULES", "Rules", "glossary", "signature"]
