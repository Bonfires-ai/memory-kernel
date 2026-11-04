# Runtime grammar keywords

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

## Value keywords

### `const(constant ← value)`

- outputs: constant; inputs: value
- kind: `value`; side effect: none; writes: nothing
- reads: nothing

Its input, a constant of the construction's meaning, as a value.

### `definite(referent)`

- outputs: none; inputs: referent
- kind: `value`; side effect: none; writes: nothing
- reads: nothing

Marks its referent's noun phrase as definite (the). No output; no effect at execution.

### `deixis(person ← role)`

- outputs: person; inputs: role
- kind: `value`; side effect: none; writes: nothing
- reads: the turn's speaker or addressee
- fails: refused before the program runs when the role is neither, or the turn has no such person (an API call with no speaker never runs a program that says me)

The turn's `speaker` or `addressee` (the role is the constant `speaker` or `addressee`), as the platform gives it: "me" is the speaker, "you" the addressee.

### `distribute(procedure, set)`

- outputs: none; inputs: procedure, set
- kind: `control`; side effect: none; writes: nothing
- reads: nothing
- fails: refused before the program runs over a set computed inside a distributed procedure

Repeats a procedure once per member of a set, the member bound in place of the set; the data ops that depend on the set or on its tool steps' results, and feed its tool steps, run once per member too (a `gather` of a member's results runs in that member's turn). A set no `distribute` scopes reaches its readers whole. Never runs, never traced.

### `either(value ← first, second)`

- outputs: value; inputs: first, second
- kind: `value`; side effect: none; writes: nothing
- reads: nothing

either(value ← first, second): the first of two results unless it is empty, else the second. Empty is nothing (None), or an empty list, set, map or string; anything else, a 0 or a false included, is a result. Both inputs are computed before it runs (a program runs every op whose inputs are bound): it chooses between two paths, it never skips one. A wh root uses it to answer from the subject's events first and from its state lines only when the events give nothing.

### `everyone(people)`

- outputs: people; inputs: none
- kind: `value`; side effect: none; writes: nothing
- reads: the turn's participants
- fails: refused before the program runs when the turn has no participants, or when one set fills both a `many` parameter and one that takes one value (say it in two clauses; teaching refuses it too)

The turn's participants, as a set (a taught body's or a live turn's procedure repeats once per member, unless every parameter the set fills, directly or through an invoked phrase's slot, takes several values (`many`): then it takes the set whole).

### `gather(values ← value…)`

- outputs: values; inputs: value…
- kind: `value`; side effect: none; writes: nothing
- reads: nothing

Its inputs, any number of values (none gives an empty list), as one list in argument order; an input that is itself a list stays one element (nothing is flattened). It runs once every input is bound. A taught phrase uses it to fill a parameter that takes several values with the results of its earlier steps that fit that parameter ("make X from them").

### `indefinite(referent)`

- outputs: none; inputs: referent
- kind: `value`; side effect: none; writes: nothing
- reads: nothing

Marks its referent's noun phrase as indefinite (a, an). No output; no effect at execution.

### `invoke(step ← construction, slot…)`

- outputs: step; inputs: construction, slot…
- kind: `control`; side effect: none; writes: nothing
- reads: the construction store (the invoked construction's live program)
- fails: refused at teach time when it would invoke itself; a repair before the program runs when the construction is no longer live (`invokes <id>, which is no longer live`) or the references nest deeper than 8

Runs a taught construction's current program as one step of this procedure; refused when it would invoke itself; repairs when that construction is gone.

### `lemma-span(lemma ← left, right)`

- outputs: lemma; inputs: left, right
- kind: `value`; side effect: none; writes: nothing
- reads: the normalised turn
- fails: a span that is not exactly one marked verb

The lemma of the one verb the normaliser marked between two offsets (the verb slot's text).

### `name-span(text ← left, right)`

- outputs: text; inputs: left, right
- kind: `value`; side effect: none; writes: nothing
- reads: the turn's text
- fails: offsets that are not integers, or a span that covers no token

The surface text between two offsets of the normalised turn (a slot's boundaries), as it was written (case and punctuation kept), with its leading determiners dropped unless a capitalised one starts a proper name ("The Brask Group" keeps "The"). A dropped quantifying determiner ("every team") makes the reading a repair before the program runs: no keyword applies the quantifier.

### `plural(referent)`

- outputs: none; inputs: referent
- kind: `value`; side effect: none; writes: nothing
- reads: nothing
- fails: refused before the program runs in a turn: 'them' refers to the results of earlier steps of a taught phrase, and a turn has none (a repair)

Marks its referent (a `window` referent: "them") as plural. In a taught phrase's body, teaching reads it as the results of the phrase's earlier steps that fit the parameter it fills, gathered (`gather`) into a parameter that takes several values (into one that takes one value it is refused), and stores no marker. No output; no effect at execution.

### `procedure(procedure ← name)`

- outputs: procedure; inputs: name
- kind: `control`; side effect: none; writes: nothing
- reads: nothing

Names a procedure (a taught body): the variable its `seq` steps and its `distribute` hang on.

### `seq(procedure, index, step)`

- outputs: none; inputs: procedure, index, step
- kind: `control`; side effect: none; writes: nothing
- reads: nothing
- fails: refused before the program runs when it orders something that is not a step of a procedure, or a step waits on a later one

Orders a step (a tool step's result, or a nested procedure) at an integer index in a procedure. The order is a contract: the first unresolved step of a procedure blocks every later step of it. Never runs, never traced.

### `window(referent ← index)`

- outputs: referent; inputs: index
- kind: `value`; side effect: none; writes: nothing
- reads: the chat's turn window

window(referent ← index): the referent of the turn `index` turns back in the chat's turn window (latest first, 0 = the latest): the entity that turn's answer named, as a Referent. The window holds the chat's last plans; a turn no root construction read (an unread turn) is not in it; plans leave it 15 minutes after their last use or past 256, and when the chat's latest plan has left it the window is empty. A turn that did not answer (a repair, tools still pending) is in it, with no referent. Refused before the program runs when the window holds no turn `index` back, or when that turn's answer named no entity or more than one (an ambiguous reference is never resolved by guessing). find-entity returns a Referent as it is, with no query (no other value passes through): it is never a name looked up again. Reads nothing from the graph.

### `within(events ← events, interval)`

- outputs: events; inputs: events, interval
- kind: `value`; side effect: none; writes: nothing
- reads: the turn's time (`at`)

within(events ← events, interval): the events, in their order, whose date falls in the interval. An item's date is its first non-empty happened_at, valid_at, told_at; an undated or unreadable item is dropped. The interval (a time filler's constant) is `D<offset>` (a calendar day) or `W<offset>` (an ISO week, Monday to Sunday), a signed offset from the turn's own day or week (`D0` the turn's day, `D-1` the day before, `W-1` the week before), counted at the fixed UTC offset of the turn's time `at` (UTC when it carries none). A date with no time is that whole day; a timestamp with no offset is read at the turn's offset. Refused before the program runs when the turn has no time. Fails on an interval that is not D/W<offset> or a turn time that is not ISO 8601. Reads nothing from the graph.

## Memory keywords

### `count(n ← set)`

- outputs: n; inputs: set
- kind: `memory`; side effect: none; writes: nothing
- reads: nothing

The number of items in a set (no query).

### `events(events ← frame, participant, role)`

- outputs: events; inputs: frame, participant, role
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)
- fails: a role outside `arg0`, `arg1`, `arg2`, `arg3`, `arg4`, `arg5` and `obj`, or a role list not parallel to the participants

The events of a frame, or of any frame of a set (`frames-of`'s output), at most 200, ordered by `happened_at`, then uuid. A participant (an entity, a uuid, or one name's candidates) narrows them to the events it takes part in; a list of such participants narrows them to the events that have ALL of them. A role is one of the closed constants `arg0`, `arg1`, `arg2`, `arg3`, `arg4`, `arg5`, or `obj` (any core role but `arg0`: a participant who is not the subject), or absent (any role); with several participants, one role for all or a list parallel to them. A participant that names nobody reads no event.

### `find-entity(entity ← name)`

- outputs: entity; inputs: name
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)
- fails: a result of nothing makes the turn a repair (the name names nobody in the graph): never an answer built from another entity

The entities (not events) whose name is the given name exactly, else the same name in any case; each as {uuid, name, kinds}. A name the turn window already resolved (a Referent, `window`'s output) is returned as it is, with no query: it is never looked up again by name, and no other value passes through.

### `identity-lines(lines ← entity)`

- outputs: lines; inputs: entity
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)

The lines of the entities' summaries that say who the entity is: copular statements with a nominal predicate whose subject is the entity (or the speaker's own first person).

### `mentions(mentions ← entity)`

- outputs: mentions; inputs: entity
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)

The messages that mention the entity: uuid, content, sender and date, in date order.

### `order(item ← set, end)`

- outputs: item; inputs: set, end
- kind: `memory`; side effect: none; writes: nothing
- reads: nothing
- fails: an end that is neither `first` nor `latest`

The first or the latest item of a set by date (its first non-empty happened_at, valid_at, told_at); undated items are dropped; None when none is dated. The end is the constant `first` or `latest` (no query).

### `recall(context ← text)`

- outputs: context; inputs: text
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's `/context` retrieval
- fails: no context callable configured, or no cached context (the repair stands without it)

The fallback: the context block today's retrieval gives for the text (first-person words resolved to the turn's speaker). The runtime runs it on a repair whose result was empty and puts it in the plan's `memory_results.recall`. In `mk serve` it reads the context caches only, never a paid call: a cache miss leaves no `recall` result and the diagnostic `recall unavailable: no cached context`.

### `role-filler(value ← event, role, except)`

- outputs: value; inputs: event, role, except
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)
- fails: a role that is neither a string nor a (frame, role) pair

The entities that fill a role of the given events, each once. The role is one role, a list of roles, or the (frame, role) pairs `wh-role` returns (a pair reads its role only on events of its frame). It never returns an entity of its `except` input: the participants the question gives.

### `slot-of-kind(value ← lines, kind)`

- outputs: value; inputs: lines, kind
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)
- fails: a kind that is not one of Person, Organisation, Artifact, Place

The graph entities of one store kind (Person, Organisation, Artifact, Place), or of any kind of a set, that the lines name, whole words in any case, longest non-overlapping matches only. The kind is a constant of the construction's meaning or the set `role-kinds` returns, each matched to the store's label whatever its case; an empty set reads nothing (no query).

### `source(source ← item)`

- outputs: source; inputs: item
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)

Where an item was said: per item, each message that mentions it (the item itself when it is a message) and the saga holding that message, with the saga's name.

### `state-lines(lines ← entity, frame)`

- outputs: lines; inputs: entity, frame
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)

The entity's summary lines about a frame. Summary lines carry no frame yet, so today it returns the identity lines (`identity-lines`).

### `time-of(time ← event)`

- outputs: time; inputs: event
- kind: `memory`; side effect: none; writes: nothing
- reads: the bonfire's graph (read-only Cypher, scoped by its group)

An event's date (its `happened_at`, else its `told_at`), or the list of them for a list of events; None for an undated one.

## Lexicon keywords

### `frames-of(frames ← lemma)`

- outputs: frames; inputs: lemma
- kind: `memory`; side effect: none; writes: nothing
- reads: the lexicon: the bonfire's rolesets (`chain/rolesets.json`) over the shipped question map
- fails: a result of nothing makes the turn a repair (the verb names no frame): never a polar no

The rolesets of a verb lemma: the bonfire's own, else the question map's, sorted; a roleset with no role is never returned. Verb-sense ambiguity travels on as the whole set.

### `role-kinds(kinds ← roles)`

- outputs: kinds; inputs: roles
- kind: `memory`; side effect: none; writes: nothing
- reads: the lexicon's question map
- fails: a role that is not a (frame, role) pair

The store kinds (Person, Organisation, Artifact, Place) of the entities that fill the given (frame, role) pairs (`wh-role`'s output), sorted: per frame, the kinds of its own roles among the pairs; a frame with none of its own roles among them takes the kinds of the modifiers every frame admits (the map's `*` entry). A role's kinds are generated from its VerbNet themes (`location`, `destination`, `source`, `initial_location` and `trajectory` are a Place; a `beneficiary` is an Organisation; the shared `argm-loc` is a `location`); a role with no such theme has none. An empty result means no kind is expected: a `slot-of-kind` over it reads nothing.

### `wh-role(roles ← frames, wh)`

- outputs: roles; inputs: frames, wh
- kind: `memory`; side effect: none; writes: nothing
- reads: the lexicon's question map

The (frame, role) pairs that answer a wh constant in the given frames, frame by frame: each frame's own roles, then the modifiers every frame admits (the map's `*` entry). A label joining two kinds (`who_or_what`) answers either; `unknown` answers none.

## Tools

Tools are not keywords. The agent runtime declares its catalogue (`ToolSpec`: name, args, `side_effect` true, false
or unknown, and the state cells it `reads` and `writes`); a construction may then use a tool like a keyword whose one
output is the tool's result. The kernel stops at the first tool step whose inputs are bound and returns it in the
plan's `tool_steps`; the plan's `licence` lists every tool the program uses with its declared cells (surfaced, never
enforced: the agent runtime's resolver enforces them).
