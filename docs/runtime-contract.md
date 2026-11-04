# The runtime contract

What the kernel hands an agent runtime (a `TurnPlan`), what the agent runtime hands back (result rows), and the rules
that bind them. The kernel plans and judges; **it never runs a tool**. The game (the agent runtime) resolves each tool
step against its world, commits the effect, and reports a verdict. The acceptance suite is
`tests/unit/runtime/test_effectful_contract.py`.

## The TurnPlan

`Runtime.plan(turn)` (HTTP `POST /v1/bonfires/{id}/plan`) returns one `TurnPlan`:

| field | meaning |
|---|---|
| `plan_id` | string: the plan's id (a uuid hex); `resume` and `retry` name the plan by it, and every `step_id` starts with it |
| `bonfire_id` | string: the bonfire the plan was made for |
| `turn` | object: the turn as given (`text`, `speaker`, `addressee`, `participants`, `chat_id`, `message_id`, `at`) |
| `status` | `answered`, `needs_tools`, `repair` or `unread` |
| `program` | the program the reading made (memory operations and tool steps), after `invoke` expansion |
| `answer_shape` | the frame's answer shape, for the answer directive |
| `answer` | the answer, when `answered` by memory |
| `memory_results` | each memory operation's result, by variable |
| `tool_steps` | the pending tool steps (below); empty unless `needs_tools` |
| `licence` | the tools the program may use, or `null` (rules below) |
| `failure` | `{step_id, tool, kind, detail}` of the step a verdict refused; else `null` |
| `trace` | the execution steps (`op`, `args`, `out`, `ok`, `reused`) and timings |
| `diagnostics` | why a plan is a repair or unread, in words |
| `reading` | how the turn was read (`applied`, `root`, `complete`, `retried`, `ms`; `alternatives` and `composed` when they apply, see [Frames](#frames) and [Context](#context)) |
| `hand` | the constructions the turn drew, with their standing |
| `frames` | what read which words of the turn, on every plan whatever its status (see [Frames](#frames)) |
| `left` | the text the primary analysis did not consume, as written (the whole turn when nothing read it) |

The trace carries `parser: "sm-fallback"` when the runtime had to parse on the small spaCy model because its configured
model would not load (see [The parser](#the-parser)). The key is absent otherwise.

A pending tool step is `{step_id, tool, args, side_effect, needs_confirmation, actor, checks}`:

- `step_id` is `<plan_id>:<i>:<occ>`. No id is valid in two plans.
- `actor` is the turn's speaker.
- `checks` is `[{param, constraint, result}]` with `result` one of `pass`, `fail`, `unknown`.

A `licence` row is `{tool, side_effect, needs_confirmation, reads, writes}`.

## Frames

Every plan carries `frames`: what read which words of the turn, from the primary analysis and the competing ones. The
agent acts on them (the kernel affords, the agent calls); the kernel runs only a complete reading through its gate.

```json
{"name": "make", "kind": "tool", "construction": "frame.make.make", "roles": {"title": null},
 "complete": false, "analysis": 0, "composed_with": null}
```

| key | meaning |
|---|---|
| `name` | a tool (a catalogue card's procedure), a taught phrase's spec id, `teach_phrase` (the editor), `yes`/`no` (a reply), or a question root's id |
| `kind` | `tool`, `phrase`, `editor`, `reply` or `question` |
| `construction` | the construction that read it: a frame part (`frame.<name>.<word>`), a card, a taught root, a base root |
| `roles` | role -> the text bound to it, or `null` when open; a tool's roles are its parameters, a phrase's its slots; the editor and a reply have none |
| `complete` | the analysis passed its goal tests (it read the whole turn) |
| `analysis` | which analysis it came from: 0 is the primary |
| `composed_with` | the open frame a follow-up composed with (see [Context](#context)), else `null` |

- **Frame parts.** Each live catalogue card and taught phrase gets one generated construction per (name, anchor word),
  as a `part` whose meaning is `(frame ?frame <name>)`. The anchor is the first verb of its form as the turn parser
  tags it, as its lemma ("please send {X} to {Y}" is `send`); a form with no verb is anchored on its first noun or
  proper noun ("the {X} report" is `report`); a form with neither ("{X} please", "who {X}?") gets no part and is read
  by its whole form only. Parts are built when the grammar loads, never stored, record no standing, and load below
  everything live, so a card still reads its own sentences completely ("take JUJ-3" is the take card, bound). A part
  applies wherever its word occurs as a whole token: "make onboard josh" reads a partial `make` frame with every role
  open, and "please take JUJ-3 and put it on Doing" reads `take` and `put`. A part never reads inside a word
  ("archive" not in "archives"), and a one-word part also reads its word as a verb ("please archive it", "it was
  archived"); two parts on adjacent plain words share a space, so only one reads.
  A word that leads a card and a taught phrase affords both. The engine applies a message's parts one after another
  (at most one part-made child per search node), so ten verbs cost ten steps, not every order and subset.
- **Roots.** The root of each analysis is a frame too: a card is a `tool` frame with its roles cut from the turn (the
  NP cut, as the tool gets it), a taught root a `phrase`, a reply root a `reply`, any other base root a `question`.
- **Up to three analyses.** A complete primary reading adds the competing analyses the engine finds
  (`RuntimeEngine.comprehend_all`), one per other root, three in all: "who is Zed?" with a taught "who {X}?" gives the
  phrase and the base memory question. The search for them leaves frame parts out (no complete analysis holds one)
  and stops at the engine's deadline inside the search, so verbs in a slot's text cost it nothing. A partial primary
  has none. When the n-best read fails, the primary stands on
  its own frames and `reading["alternatives"]` says why. An analysis through a draft not usable here affords nothing.
- **Inside a bound role.** A slot's filler can swallow another verb's clause: "search about the monkey situation and
  write note about it" reads completely as `search` with the whole rest as its query. A turn with two marked verbs or
  more reads each bound role of its complete frames on its own (one level deep), and the `tool`, `phrase` and
  `editor` frames found there follow the frames above (`write(body: note about it)` here): an affordance, never an
  action (C5).
- **The editor and the replies are base rows**, never an office catalogue row: `frame.editor.teach`,
  `frame.editor.mean` and (1.5.2) `frame.editor.phrase` (parts of "teach", "means" and "phrase", reading "teach",
  "X means Y", "file X means ...", "I mean" and "make a phrase for that" anywhere) afford `teach_phrase`; `reply.yes` and
  `reply.no` (one row more per final "." or "!") are narrow roots. A reply plan is `unread` with its reply frame:
  it answers the turn before, nothing runs. "yes please take it" is no reply.
- **No catch-all.** No construction completes any message: a message nothing reads has no frame and is `unread`. An
  `{X}` root would read every turn completely and silence every frame.
- **A partial reading is never run.** It is `unread` with its frames; the agent decides what to call.

## Context

`Runtime.plan(turn, context=[Frame])` (HTTP: a `context` list in the plan body) names the open frames of the
speaker's turn before: a frame whose effect has not run (the caller never sends a frame whose call committed), with a
role still open or held back by the agent (bound, but it asked instead of calling). At most 8 frames; each needs a non-empty `name`, a `kind` of the five and `roles` (role -> text or
`null`); `construction`, `complete`, `analysis` and `composed_with` may come with their types; any other key, or a
wrong type, is refused (HTTP 422).

- **Composition.** A turn that reads as a bare fragment (nothing complete, and no frame of its own) is read again
  inside each open frame's forms: for each context frame of kind `tool` or `phrase`, each live root it names (the
  tool's cards, or the phrase), each slot whose role is open, that root rendered with the turn's text in the slot and
  the frame's bound roles in the others ("JUJ-3" against an open `take` frame is read as "take JUJ-3"). The turn's
  final marks are dropped first: they end its sentence and would land mid-way in a rendering ("JUJ-3." against
  give(person=Milo) is "give JUJ-3 to Milo"). A closing quote or bracket (spaCy morph `PunctType` Quot or Brck) is
  not a final mark and stays with its opener ('"JUJ-3"' binds '"JUJ-3"'). A turn of nothing but marks ("?", "...")
  composes with nothing: it stays an unread turn. A root with
  another slot still open is not rendered ("Doing" against `put` with the ticket open composes nothing).
- **A frame held back is re-filled, after every open role.** A frame whose one bound role is the slot of a one-slot
  root is then rendered with the turn's text in place of that value: "take the new one" reads completely as
  take(ticket="new one") (a ticket can be named by its title), the agent asks "which ticket?", nothing runs, and
  "JUJ-3" is read as "take JUJ-3". A frame with two or more roles bound is never re-filled (which value the answer
  replaces is the agent's to say): "Doing" against put(ticket=JUJ-3, place="new lane") composes nothing.
- **The first rendering that reads completely through a root of that frame is the plan's reading.** It runs as that
  root's reading would; its frames of the open frame's name carry `composed_with`; `reading["composed"]` is
  `{"with": <name>, "text": <the rendering>}` (or `{"error"}` when the engine failed it, the turn's own reading then
  standing); `reading["ms"]` counts every read. `turn.text` stays the turn's own text.
- **Never composed:** a complete turn ("look at the inbox" ignores an open `take`), and a turn with a frame of its own
  ("make onboard josh"): the whole message never fills an open slot. With no context, "JUJ-4" reads nothing.

### The `/context` frame

`POST /v1/bonfires/{id}/context` (`mk serve`) returns a `frame` read from the query by the wh-word rules (unless the caller passes `answer_shape`, which sets the shape): no Jev call,
no frame cache, no query shard. It carries `shape` only.
cached by an older kernel is never returned. A client must not read either key.

## Teaching

`Runtime.teach(TeachRequest)` (HTTP `POST .../teach`) compiles a definition into a staged draft; `confirm(draft_id,
by)` (HTTP `POST .../teach/{draft_id}/confirm`) makes it live. Teaching is admission: a teach reads its body clauses
and its forms' own renderings, a handful of reads (under 2 s), and holds the bonfire's engine only for that.

- **Refused** (a draft with `problems`, nothing staged): a malformed request; a body clause that does not read, or
  reads as a question, a quantified phrase or the turn window; a reference ("it", "them") that names no one earlier
  step; a program the registry refuses; a step that runs in the agent after an office step that waits for the board's
  approval (see [Structured steps](#structured-steps)); a form that does not read its own renderings through itself
  once loaded on top of the grammar (the self-read); a form with the skeleton of a live construction (its normalised
  literals, every slot one placeholder) unless `replace` is set, and with a base or catalogue construction even then.
- **Duplicates.** A draft every form of which is already a live taught phrase or catalogue card (the same normalised
  form and the same meaning up to slot and variable names) is refused with one problem per form naming it, and
  nothing else: `form 'file {X}' is already the live construction <id>`. A draft only some of whose forms are live as
  they are (a phrase taught again with `replace` and a new default for its short form) is taught as any other.
- **No generalisation refusal.** A broad form ("who {X}?" next to the base `who-be-x`) is admitted. Where it and
  another construction both read a turn, both reach the agent as competing analyses (see [Frames](#frames)), and the
  outcome of the call the agent makes moves their standing. Wider overlaps are never searched for at teach time.
- **`competes`.** The draft reports, per form text, the frames the live grammar reads that form's own renderings as
  before the draft loads: "look at the office" with the office catalogue gives the `look` card, `place` "office". A
  form nothing reads gives `[]`. The agent tells the teacher what the phrase will compete with.
- **Replace** removes, at confirm, the live taught or mined rows with the draft's skeleton (recorded at teach time in
  `provenance["replaces"]`, and at confirm in `provenance["replaced"]` with the successor). A catalogue card is
  never replaced by teaching.
- **Confirm reads nothing.** It checks the teacher, the drafts the body read through (confirmed first), and a live
  construction with the draft's skeleton gone live since (refused: teach it again with `replace`).
- A staged draft loads below everything live and is usable only by its teacher, in the chat where it was taught.

### Structured steps

1.5.2: a body entry may be a call instead of a clause: `{"call": "<tool>", "args": {"<param>": "<value>"}}`. Any tool
of the catalogue composes this way, one with no `x-forms` too (an agent's tool, `think`). Admission is the same: the
self-read of the forms and the duplicate check; nothing in the call is read by the grammar.

- **A value** is one of:
  - `"{X}"`: a slot of the forms. A slot the form leaves out is its default's text, as written (a literal: "me" stays
    "me", where a clause reads it as the speaker).
  - `"it"`: the result of the step just before, which must produce one that fits the parameter (see [References in a
    taught body](#references-in-a-taught-body); an untyped result fits any parameter). Only the exact word is a
    reference: "summarise it in three lines" is a literal.
  - `"them"`: what it is in a clause: every fitting earlier result, into a `many` parameter only.
  - Any other string: a literal.
  - A value that holds a brace but is not exactly one slot ("news about {X}") is refused.
- **Every parameter the tool declares is given.** A tool step takes all of its parameters, as a form binds all of them;
  a parameter left out, one the tool does not have, an unknown tool or a keyword (`recall`) is refused, naming it.
- **It compiles to the ops a read clause gives**: `procedure` + `seq` + the tool op, its parameters in declared order,
  each literal a `const` op. A slot that fills the tool's `x-returns` parameter rebinds as in a clause.
- **Shape.** `call` a non-empty string, `args` (optional when the tool has no parameter) an object of non-empty strings,
  nothing else. HTTP: a body entry of another shape is a 422; a tool or parameter name holds at most 256 characters, a
  value at most 500, a call at most 32 parameters.
- **An agent step after the board (C12).** A step whose tool runs in the agent (`x-runner: agent`) after an office step
  that waits for the board's approval (a side effect the speaker's turn does not consent to: `x-consent: confirm`, or an
  unknown effect) is refused, in the expanded program (through an invoked phrase too): the board's approval runs the
  office's step, and nothing would run the agent's after it. An office step after it, or an agent step before it, is
  admitted.
- The request is kept as sent (`provenance["request"]["body"]`); the draft id does not depend on a call's key order.

## The parser

The runtime parses every turn once, with one spaCy model, whatever the turn is (a question, a command, a taught
phrase). The model is chosen per purpose:

| purpose | setting | default |
|---|---|---|
| `turn` (everything said in a turn) | `MK_TURN_SPACY_MODEL` | `en_core_web_trf` |
| `extraction` (build, append and the parser service) | `MK_EXTRACTION_SPACY_MODEL` | `en_core_web_trf` |
| `query` (serve and the runtime's memory ops reading graph text back) | `MK_EXTRACTION_SPACY_MODEL` | `en_core_web_trf` |

- **Subject frame.** A turn with no subject that does not open on a verb is read as addressed ("you ..."), so a bare
  command parses verb-first. The frame word is never the actor (the runtime decides the actor), and the parse is
  returned over the original text with its original offsets.
- **Fallback.** If the `turn` model other than `en_core_web_sm` fails to load (not installed, a missing package or
  factory), the runtime logs a warning, parses on `en_core_web_sm` (the frame still applies) and sets the TurnPlan trace
  note `parser: "sm-fallback"`. A turn never fails over the parser. `extraction` never falls back: it fails loudly,
  because a silent fallback would build a different graph.
- **Query policy.** Serve never goes empty over a missing model. The readers that run at query time (the `/context`
  entity sheet's identity lines, the runtime `identity-lines` op, the tense reading behind relative-time rendering) use
  the `query` purpose: the configured extraction model, and when it will not load (a bare install without
  `memory-kernel[parser]` or the trf model) `en_core_web_sm`, with a logged warning and a note (`parser_note("query")`
  is `"sm-fallback"`). Only a missing `en_core_web_sm` as well leaves them with nothing to read; that skips the reading
  and never raises. Three policies, then: build fails loud, query and turn fall back with a note.
- **One copy per model.** A process loads each spaCy model once, whatever the purpose: `turn`, `extraction`, `query`,
  the grammar operations and the event stages all share it (guarded by a lock, so two threads never each load one). A
  process that used to hold three transformer copies (about 2.5 GB resident) holds one (about 1.2 GB). NER is off in
  that copy; nothing consumes entity output (the attribute is read but always empty).
- **Whitespace.** An empty or whitespace-only text (`" "`, `"\n"`, `"\xa0"`) is tokenized and not parsed, because the
  transformer model raises on it; every parse entry point does this. A turn is parsed stripped and returned over the
  text as written, with the leading and trailing whitespace as space tokens and every other offset exact, so `" hum a tune"`
  reads as `"hum a tune"` does.
- **Extraction on trf.** Extraction defaults to `en_core_web_trf`; `MK_EXTRACTION_SPACY_MODEL=en_core_web_sm` opts out.
  It needs `memory-kernel[parser]` and the trf model (`mk setup` installs it) and fails loudly without them. The
  closed-class anchors are one set for every parser and do not change. On CPU a build's projection stages take
  noticeably longer (a 1,778-message build spent about 27 minutes more in build_dry and write); `MK_SPACY_GPU=1` with
  the `gpu` extra moves it to a GPU, and a parse-once fix is planned for 1.4.1.
- **Install.** `mk setup` installs the pinned `en_core_web_trf` 3.8.0 wheel (436 MB) with the other components. The
  model loads only with `memory-kernel[parser]` installed (`spacy-curated-transformers`, torch, transformers), and
  `mk setup --check` reports it not OK when it cannot load.
- **Speed.** A turn parses in about 90 ms on CPU. Optional GPU: `pip install "memory-kernel[gpu]"` adds
  `cupy-cuda12x>=13` (CUDA 12), and `MK_SPACY_GPU=1` makes the process call `spacy.prefer_gpu()` once before its first
  model loads (logged; with no usable GPU spaCy stays on the CPU). The extra alone changes nothing.

## Result rows and verdicts

`Runtime.resume(plan_id, results)` (HTTP `POST .../plan/{plan_id}/resume`, body `{"results": [...]}`) takes one row per
pending step:

```json
{"step_id": "...", "tool": "...", "args": {}, "verdict": "committed", "value": null, "detail": null}
```

`verdict` is one of `committed`, `precondition_failed`, `tool_error`, `declined`. `committed` carries the `value` the
world produced; the others carry a `detail`.

A row whose `tool` or canonical-JSON `args` differ from the pending step is rejected (`ResumeError`, HTTP 409) before any
state changes. Key order in `args` is moot. A refused verdict makes the plan a `repair` with `failure` set; the plan's
licence is kept. A repair waits on no tool, so a resume against it is a `ResumeError`.

## Step ids, retry and committed effects

- **Retry.** `Runtime.retry(plan_id)` (HTTP `POST .../plan/{plan_id}/retry`) re-issues only the failed step, under the
  same `step_id`; earlier committed steps stay committed. The plan returns to `needs_tools` with `failure` cleared. A
  plan with no `failure` is a `ResumeError`. A retry records its own outcome on the plan's constructions.
- **Committed.** `Runtime.plan(turn, committed=[{tool, args, value}])` (HTTP: a `committed` list in the plan body) names
  effects already done. The `args` must equal the step's `tool_steps[].args` exactly (entity dicts included). Each entry
  resolves one step: it is answered with the entry's value (trace `reused`) and never issued. Every entry needs a
  `value` key (`null` when the effect produced none); an entry without one is refused (HTTP 422). Entries with the same
  `tool` and `args` resolve in the order given, so list `committed` in the order the steps were issued (ledger order).
  A phrase run once per member issues identical calls (`take {"item": "water"}` for each member), and a result flows
  into later args: out of order, one member's result binds to another and an effect already done is issued again.
- **Replay under composition.** A step whose args hold an earlier step's result is issued again with identical args
  when that step is committed: the committed `value` binds exactly as given, a structured or non-ASCII value and the
  list a `gather` builds included, and a canonical-JSON comparison of the args holds. Plan the same turn again with the
  steps already run in `committed`, in issue order; once the last one is committed nothing is issued. A phrase run once
  per member replays each member's own results.

### Recovery

A plan is kept for 15 minutes after its last use, and a catalogue reload drops every plan of the bonfire. A `resume` or
`retry` of a dropped plan answers `404` ("the plan expired or the catalogue was reloaded"). The game then plans the same
turn again with `committed`: the steps it already ran, so the new plan issues only what is left.

## The licence

`licence` is `null` only for an `unread` turn, or a repair of a program that names no tool (a memory question, which the
agent runtime hands to today's path). A program that names any tool, or contains an `invoke`, carries its own licence on
every outcome: a plan-time repair, a verdict repair, a repair after a committed effect. A dangling `invoke` repairs with
`[]` at least. A tool a taught row copied inline that the catalogue has since dropped is still listed, with
`side_effect: null` and `needs_confirmation: true`. The kernel surfaces the licence; the game enforces it.

## Slot constraints

This corrects amendment G3 ("an unmet constraint the kernel can see repairs at plan time"): only a store-kind mismatch does.

A catalogue parameter may declare a `constraint` (a WordNet concept, such as `tune`) or a `kind` (a store kind, such as
`Person`):

- A lexical constraint **never blocks**. Its result rides on the issued step as `checks` for the game's Resolve (the
  WordNet rule is coarse: any sense under the concept passes).
- Only a store-kind mismatch on a resolved memory entity blocks: a `repair` before any step, with the licence kept.
- Today this applies only to hand-written cards whose meanings resolve entities (`find-entity`). Generated catalogue
  cards bind slots as text and never resolve entities, so a catalogue `kind` does not block. Whether kind-typed
  parameters resolve is for the workshop plan to decide (R7).

## Standing

World verdicts (`precondition_failed`, `declined`, `tool_error`) count in a construction's `uses_world` only. They are
neutral for precedence and never lower `uses_failed`. A misreading, an empty result or an unresolved name counts as a
failed use. A `committed` end of turn counts as `uses_ok`.

## The confirmation policy

`needs_confirmation` on a step and a licence row comes from `Runtime(policy=...)` (default `DEFAULT_POLICY`):

1. `side_effect` false: never.
2. `side_effect` unknown (`None`): always.
3. `consent: "speaker_turn"` and the step's `actor` is the turn's speaker: not marked (the turn is the consent).
4. Otherwise: marked.

## `invoke`

A taught body clause that reads through a taught construction stores a **reference** to it, not a copy.

- It expands at plan time to the live version, following `provenance["replaced"][].successor`. Nesting is at most 8
  deep.
- A cycle is refused at teach time and repaired at plan time.
- A confirm that removes a row with no successor (a replace by generalisation a 1.5.0 draft recorded; a 1.5.1 replace
  always has one) orphans its invokers: the confirm names them in `provenance["warnings"]` of the confirmed rows
  (HTTP: `warnings` in the response).
- A dangling reference repairs with its name, and the invoker records `failed`.

## References in a taught body

A clause of a taught body may refer to an earlier step's result: "it" (one result) or "them" (all of them). The kernel
binds the reference when the body is taught; a stored body never holds the turn window.

- **Producers** are the earlier tool steps of the same body, in body order: each a clause of exactly one catalogue
  tool. An `invoke` clause, or a clause of several tools, is not a producer.
- **Fit.** A producer fits a consuming parameter when that parameter has no `kind`, or the producer's kind (the `kind`
  of its tool's `x-returns` parameter) equals it. A tool with no `x-returns` gives an untyped result (every agent tool
  and `think` does).
- **Untyped results fill string parameters (1.5.2, C13).** Where no earlier producer has a kinded parameter's kind, an
  untyped result fits it too: "web search for X, think: pick the most urgent, make a ticket from it" composes, and the
  tool checks the value when it runs (the office refuses a ticket that does not exist). Producers of the kind come
  first, so a body 1.5.1 accepted binds as it did ("take X, look at Doing, give it to R": "it" is take's ticket). A
  live plan never refuses a step for a text value in a kinded parameter (only a resolved entity of another store kind
  blocks, see [Slot constraints](#slot-constraints)).
- **In a structured step** "it" is the result of the step just before (see [Structured steps](#structured-steps)), not
  the one fitting producer.
- **Singular.** "it" must have exactly one fitting producer; with none the teach is refused. With several it is refused
  ("say which (use a slot of the form, or 'them' for all), or write the step as a structured step"), unless every one
  of them is untyped (its tool has no `x-returns`: an agent tool, `think`): then "it" is the nearest of them, as a
  structured step's "it" is the step just before (1.5.2, C13). So "web search for X, think: pick the most urgent, write
  it" writes what think gave, and "... make a ticket from it" makes a ticket of it. A producer whose tool has an
  `x-returns` is never passed over for a nearer one: "take X, think: say why, write it" is still refused. Into a `many`
  parameter "it" is a list of one: a reference ("it"/"them") into a `many` parameter is always a list. Plain slot or
  constant text into one stays text, as in 1.4.
- **Plural.** "them" fills only a parameter that takes several values (`many`), with the list of every fitting
  producer's result in body order (a list of one when there is one). Into a parameter that takes one value it is refused.
- **Rebinding.** A slot that fills the `x-returns` parameter of a tool refers, in later clauses, to that step's result:
  in `take water`, `make {X} from them`, `give {X} to {R}` the give gets what make returned, while make itself gets the
  slot's text (a later step that returns the same slot gets the earlier result). A constant never rebinds; a slot
  produced twice means the latest producer. Without `x-returns` no slot is rebound; "it"/"them" still bind to untyped
  results where the parameter has no `kind`; bodies 1.4 accepted compile unchanged.
- **Once per member.** A step that runs once per member (a clause `everyone` distributes: "give tea to everyone") has
  several results.
  - A reference to it is refused ("runs once per member, so it has several results").
  - A slot it fills is not rebound: later clauses keep the slot's text.
- **No earlier step.** A reference in the body's first clause (no earlier clause at all) would read the turn window:
  refused, the message saying a body's referent must be "a slot or a constant, or an earlier step's result of this
  phrase". After only an `invoke` clause or a clause of several tools, the message is "no earlier step of this phrase
  produces one that fits" the parameter.
- **`everyone` into both kinds.** One `everyone` set read both into a `many` parameter and into one that takes one value
  is refused: a taught "serve {R}" = `make tea from {R}`, `give cake to {R}`, then "serve everyone". Teaching checks the
  expanded body, and a live turn is a repair. Both say "'everyone' fills both a parameter that takes several values
  ('material') and one that takes one ('person'): say it in two clauses".
- **`everyone` through an invoked phrase.** Whether `everyone` distributes, and whether it fills both kinds, is decided
  by what it feeds once invoked phrases are expanded, in a taught body as in a live turn: with "stock {R}" = `make tea
  from {R}`, a taught body "stock everyone" makes once, with all the participants as the material.
- **A live turn.** "them" outside a taught body ("give them to Ana") is a repair, not an error: 'them' refers to a
  taught phrase's earlier steps, and a turn has none. An `everyone` set distributes a live turn's procedure once per
  member, except when every reader is a `many` parameter, where it stays one list.

## The catalogue

A bonfire's catalogue (`BonfireMeta.catalogue_json`, absent until the first `PUT`: no tools) is `kernel-catalogue/v1`. Each tool has:

- `name`, `input_schema` (its properties are the parameters) and `x-side-effect` (required, boolean).
- `x-args`: one entry per parameter with `role` (required) and optional `kind`, `constraint`, `many`.
- `x-consent`: `confirm` (default) or `speaker_turn`.
- `x-runner` (optional, 1.5.2): `office` (default) or `agent`, where the tool's step runs. The kernel keeps it
  (`ToolSpec.runner`, the registry's `OpSpec.runner`) for the teach-time rule of [Structured steps](#structured-steps);
  the caller routes each pending step by it. Any other value is refused naming the tool.
- `x-returns` (optional): `{"of": "<param>"}`. The tool's result stands for the referent of that parameter (`make`'s
  result is the made `item`); its kind is that parameter's `kind`, if any. The parameter must exist and must not be
  `many`. Without it the tool returns an untyped value. See [References in a taught body](#references-in-a-taught-body).
- `x-forms`: forms such as `"play {pattern} on {source}"`; every parameter is bound by a slot.

Each form makes one live hand card, `card.<tool>.<n>`. Cards are attached in memory beside the stored rows and never
persisted.

- `GET /v1/bonfires/{id}/catalogue` returns `{sha256, tools}`, `tools` being the stored catalogue's tool rows, read
  now. `PUT` stores a validated file (400 with the reason on a bad one) and answers with exactly what a `GET` returns
  after the install; it does not move `committed_run`. `mk catalogue install BONFIRE_ID FILE` does the same from the CLI.
- One gate decides "would this load?": a file is installed only if the runtime's registry takes its tools and every card
  compiles. A tool named like a keyword (`recall`, `answer`) or a reserved value op (`gather`, `plural`), a name outside
  the construction-id charset (`a-z`, `0-9`, `.`, `_`, `-`, compared lowercased) and two names that differ only in case
  are refused (400, `mk catalogue` exits 2), and the previous catalogue keeps serving.
- A tool name must match the construction-id charset even when the tool has no `x-forms`. The gate is strict on purpose:
  tool names are program ops.
- A `PUT` drains in-flight calls within `MK_SERVE_TEACH_WAIT_MS`, then reloads the bonfire's runtime. A cold engine start
  counts as in flight; a start longer than `MK_SERVE_TEACH_WAIT_MS` (15 s default) gives the `PUT` a `503` ("busy ...
  retry later") and the catalogue is unchanged. In-memory plans are dropped: an old plan id answers 404 (see
  [Recovery](#recovery)).
- A plan made while the bonfire teaches waits up to `MK_SERVE_TEACH_WAIT_MS` (default 15000), then answers `unread`
  ("runtime busy: teaching", or "runtime busy: reloading the tool catalogue" while a `PUT` reloads it).
