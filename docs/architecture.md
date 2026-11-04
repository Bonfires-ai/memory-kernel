# Architecture

How `memory-kernel` is put together: what `mk build` does stage by stage, what `mk ask` reads, how a build knows which
code made it, where the runtime data and the parser come from, and how the runtime grammar reads each incoming turn
(§8). This document describes the code as it is in this
tree; the release notes are in [CHANGELOG.md](../CHANGELOG.md).

## 1. Overview

```text
conversation JSON
      |  mk build            14 stages, outputs under <workdir>/<sample_id>/
      v
conversation root  ----------------->  Neo4j group (typed events, participants, relations, search keys)
      |                                         ^
      |  mk ask "question" --group G            | read-only
      v                                         |
query frame -> retrieval over the group -> answer context -> short answer
```

The build is a construction-grammar pipeline. The conversation is parsed, a grammar engine (FCG Go, run as a local
subprocess) turns each message into candidate events with typed roles, language-model passes decide the episode boundaries and
the eventuality of each predicate, the result is projected into graph rows, and the rows are written to one Neo4j group. Retrieval (`mk ask`) classifies the
question into a frame from its wh-word rules, plans a search over the group's keys and structure, assembles a bounded context of evidence lines
and asks a chat model to answer from that context only.

The code is laid out as:

| path | role |
|---|---|
| `memory_kernel/cli.py` | the `mk` command: `build`, `ask`, `inspect`, `setup`, and a hidden per-stage child |
| `memory_kernel/pipeline/` | the build driver: `runner.py` (order, markers, resume, spend stops), `config.py` (`BuildConfig`), `ledger.py` (the spending shard), `llm.py` (senders and the stage meter), `engine.py` (FCG Go and parser lifecycle), `provenance.py` (code identity), `graphdb.py` (Neo4j), `frames.py` (PropBank frame dirs), `embed.py` |
| `memory_kernel/pipeline/stages/` | one function per stage (`STAGES[name](ctx, live)`), grouped in `ingest`, `parse`, `graph`, `events`, `write`, `receipt`, `index` |
| `memory_kernel/serve/` | `answer.py`: `mk ask`; `http.py`: `mk serve`; `runtime.py`: the runtime grammar's routes and engines |
| `memory_kernel/runtime/` | the runtime grammar (§8): programs, keywords, the executor, its own FCG engine, constructions, teaching, `plan` / `resume` |
| `memory_kernel/setup_runtime.py` and `data/runtime-pins.json` | `mk setup` and the pins it enforces |
| `memory_kernel/parser_service/` | the syntactic parser that the capture stage starts |
| `memory_kernel/construction/`, `graph/`, `search/`, `grammar/`, `knowledge/`, `envelope/`, `windows/`, `ingest/` | the construction-grammar, graph-projection and retrieval libraries that the stages and the serve step use |
| `memory_kernel/data/` | package data: the stopword list, the closed-class anchors, the pronouns and identity constructions, the runtime pins and runtime grammar, the PropBank annex |

## 2. The build

`pipeline.runner.ORDER` is the stage order: the 14 stages `mk` runs, `select` to `seal`: `select`, `boundary`, `capture`, `graph_picks`, `graph_a`, `events`, `ctx`, `referents`, `build_dry`, `write`, `rewrite`, `index`, `sanity`, `seal`. `--through STAGE` stops after a stage;
the default is `build_dry`. Evaluation stages are not part of `mk`; `mk inspect` lists only these 14.

| stage | output dir | what it does |
|---|---|---|
| `select` | `01-select-001` | splits the input into a question-blind conversation and the evaluation questions; prepares the message rows and sources. No provider, no Neo4j. |
| `boundary` | `03-boundary-001` | boundary windows over the sources, decided by the Jev model (policy `walk`, the default) or, under `--boundary segmenter`, embedding cuts per session with no Jev request (see Boundary policies). Creates the ledger shard `03-boundary-001-ledger`. Has an opt-in replay gate (`--boundary-replay-archive`). |
| `capture` | `04-syntax-001`, `04-capture-001`, `05-runs-001` | the only stage that holds the FCG Go engine: syntactic parse capture of every message through the local parser, then the engine's comprehension runs. $0. Resets the engine's id counters before every job, and can run on several engine shards (see Parallel capture). |
| `graph_picks`, `graph_a` | `06-graph-001` | the whole-conversation event graph: antecedent picks, then graph A. |
| `events`, `ctx` | `10-events-001`, `10-eventuality-ctx-001` | eventuality per roleset, then per predicate in context. |
| `referents` | `10-referents-001` | an append's decisions on specific mentions ("the kiln"): the common ground of each stored candidate, an easy join, or a Jev yes/no for a hard case; a new specific referent is always `kiln#<8 hex>` (the plain key names the kind; pre-split plain entities, `legacy` in `ground.jsonl`, stay candidates); writes `decisions.json`, which `build_dry` and `write` project with. A full build writes a marker and asks nothing. |
| `build_dry` | `11-dryrun-001` | the projection of the conversation into graph rows, validated against the row contract and the invariants. No Neo4j. |
| `write` | `11-write-001` | writes the rows into ONE empty Neo4j group. |
| `rewrite` | (in `11-write-001`) | an idempotent re-write of the same rows. |
| `index` | `11-index-001` | persists the group's search-key index (embeddings). |
| `sanity` | `11-sanity-001` | a receipt, no search, no provider, $0: a full build's is `mk-build-receipt/v1` (the index's message count, recipe tag and stamped keys); an append's is `mk-append-receipt/v1` (gates, what the projection minted, reused and routed, referents). |
| `seal` | `11-seal-001` | hashes the conversation root. |

Stages that call a paid provider are the ones with an entry in the driver's cost table (`INDEX_USD`): `boundary`,
`graph_picks`, `events`, `ctx`, `referents`, `write`, `rewrite` ($0 in the table) and `index`; `sanity` has a table entry of $0 (a receipt, no provider). The ten reference conversations cost $2.36 to build in 1.8 against $8.21 in 1.7.

How the driver behaves:

- **Markers and resume.** A finished stage writes `.done-<stage>` in its directory; a re-run skips it. A started but
  unmarked stage stops the build unless the stage is resumable (`capture`, `events`, `ctx`). `boundary` is not: its directory is created exclusively.
- **Stages in the driver's process.** A stage runs inside the process that drives the build (`mk build`, or the kernel
  worker for an append), with a fresh stage context each time (`pipeline/inprocess.py`). The stages in
  `inprocess.CHILD_STAGES` run as a `python -P -m memory_kernel.cli _stage ...` child instead, each for a reason written
  next to it: `capture` imports `pyfcg` at runtime and installs a signal handler, which fails off the main thread. Both
  paths write the same result file and import audit, and an exception in an in-process stage stops the build naming the
  stage and the exception type. One build per process runs stages in-process at a time: a build that finds the lock held
  runs every stage as a child (`in-process-busy` event), and `MK_STAGE_CHILDREN=1` does the same on purpose
  (`stage-children` event). A child gets secrets through its environment, never through a file; an in-process stage uses
  the driver's config. `mk _stage` still runs one stage by hand in its own process.
- **Events.** The driver appends one JSON line per event to `<conv root>/driver-events.jsonl`; `mk inspect` reads it.
- **Money.** `--cap-usd` is checked before paid stages, a per-conversation stop applies, and the Jev ledger shard has its own
  fail-closed cap. The cost ledger is `<workdir>/cost-ledger.jsonl`.
- **Preflight.** Before any stage, the driver collects every problem it can find without spending: a missing frame dir or
  engine, a missing parser model, a bad session date, a Neo4j group that is not empty (for the first write), an unreachable
  Neo4j, a missing WordNet corpus when any stage after `select` is pending, and no
  Voyage source for a pending stage that embeds text: for `events`, neither a Voyage key nor its own cache file
  (`logs/10-events-voyage-cache.json`); for `write`, `rewrite` and `index`, neither a Voyage key nor
  `--embedding-cache`. A cache is checked for presence only, so a text it lacks still stops the stage when it runs unless a
  key is set. A `write` also embeds the Saga centroids' missing texts, so a walk write whose events cache lacks some texts
  needs the Voyage key (an `--embedding-cache` alone may be refused at write). It refuses with all of them at once. A missing OpenRouter key is not a preflight problem: the first stage that
  sends a request stops.
- **Write integrity.** `write` checks only its own group, so other tenants writing a shared Neo4j at the same time never
  fail it: the writer refuses a row uuid that another group holds (`kg_writer.ForeignUuids`, naming that group) or an
  edge endpoint outside the batch and the group, and an append reads back the rows it wrote by uuid and fails when
  one is missing. A uuid is a global
  identity: a bonfire whose chain was minted under another group (a copy that kept its source's uuids) is refused
  before `mk append` or `relocate` does anything (`forward.check_chain_group`). The write issues the kernel's own uuid
  indexes (`IF NOT EXISTS`) beside Graphiti's. Under `BuildConfig.strict_db_snapshot` (the bench's strict mode) the
  write also snapshots every group and fails if any other group changed while it ran; `--concurrent-group-prefix PREFIX`
  then tolerates *growth* of other groups whose name starts with PREFIX (another build writing at the same time) and
  nothing else; a shrink, or any change to nodes and edges without a group, always fails.

### Boundary policies

The policy is a per-bonfire build option (`--boundary`), recorded in `build-config.json` only when it is not `walk`;
an append inherits it. `walk` is the Jev topic walk described above and is unchanged. `segmenter` is meant for talks and
documents: each session is cut by the long-form segmenter (target 60 messages per episode, ceiling 150, time gaps of
12 hours on), every message lands in exactly one episode, with no Jev request. It embeds with Voyage voyage-3, prefilling every message
text into the `events` stage's cache so that stage pays nothing again; the cost is a `voyage` line on the `boundary`
cost row and `--cap-usd` stops it. An append to a `segmenter` bonfire writes each new session as new Sagas and continues
the numbering.

Under both policies the `write` stage stores on every Saga its centroid: `centroid_embedding` (L2-normalised mean of the
messages' voyage-3 vectors), `centroid_model` (`voyage-3/document`) and `centroid_messages`. The vectors come from the
shared cache; only missing texts are embedded, as a `voyage` line on the write's cost row. Nothing reads the centroids yet.

### Entity keying and kinds

`referent_policy` (`phrase` | `structure`) is a per-bonfire build option, recorded in `build-config.json` only when it is
not `phrase`; an append inherits it. `phrase` calls the same functions as before the option existed. `structure`
refuses to run without WordNet, in runner preflight and in `project`.

- **Mentions** (`graph/g1_mentions.py`). A role filler is decomposed on the projection's own parse, using the capture's
  token indices when they align with it, else the token run of the filler text in the unit nearest them (`realign`:
  characters exact with whitespace removed, so blind to tokenisation; no frame cuts, the unit's other capture indices
  being suspect too), and the filler text's own parse otherwise; on that own parse a mention needs a nominal head (NOUN, PROPN,
  PRON) or a name. The decomposition removes the tokens of a nested
  frame, splits conjuncts, and keys a name, or possessor + compounds + head (`_key_tokens`: a 3rd-person pronoun
  possessor is dropped, by spaCy morph; a nominal one keeps its own key without determiners; a 1st/2nd-person one
  stays for `owner_key`). A name compound of a common noun ("the
  Zorvex spyware") is its own referent and the name inside it an inner-name mention, never its key; a lowercased known
  name ("the brask labs") is a name. A partitive ("this kind of X") is replaced by X when its head's first noun sense is
  `noun.quantity`, or is `kind.n.01` or a hyponym of it (WordNet hypernym closure), with det, compound and
  one prep dependents (a compound name on it is an inner name); other cognition nouns are not partitive. Names inside a core are returned as mentions only. A bare NOUN head (`bare_light`: first noun sense in
  `noun.quantity`) whose dependents are all in `_LIGHT_DEPS` (det, relcl, acl, punct, cc, conj; any other dependent, a
  complement, blocks it) refers to nothing. Only a NOUN head gets a head lemma (`Mention.head`), so an adjective
  ("rich") has no class.
- **Kinds** (`graph/g1_kinds.py`). Each filler adds `KindEvidence`; the evidence of all mentions of a referent is
  summed and `decide` picks one kind in order: Person; a supplied kind noun (a predicative noun for the PPT role only,
  an appositive head, a common noun before a name head); the referent's own head-noun class; Place when a location-type role and
  a location-compatible class both hold. `g1_names.KIND_OF_LEXNAME` maps `noun.person`, `noun.group`, `noun.artifact` and
  `noun.communication`, `noun.location` to Person, Organisation, Artifact and Place; a noun takes its first sense's kind
  unless the first two senses disagree. A `noun.group` sense gives Organisation only when the first noun sense is
  `g1_names.ORGANISATION_SYNSET` (`organization.n.01`) or below it in the hypernym closure; otherwise no kind.
- **Evidence.** A `structure` bonfire stores each entity's evidence in its graph. An append
  recomputes the kind of a stored entity from it and flags the row `kinds_replace`; the writer then removes the kind
  labels the row does not carry. This is the one exception to frozen rows, and `phrase` labels never take it.
- **Events** (`graph/g1_projection.frame_name`). Under `structure` an Event's name is its frame and core roles; the
  sentence is `source_text`, shown in answer context when present (timeline and support lines; the index reader
  returns it). The Event's `key_text` is unchanged. Entity names are per referent, and message search keys hold them,
  so a `structure` bonfire's search keys differ from a `phrase` build's.

### Capture resume rule

A capture that is COMPLETE is read as is, whoever made it. An INCOMPLETE capture cannot be resumed by a different install:
its manifest seals the code pins of the run that made it, the parser address and the snapshot, and every resume re-checks them,
so a change shows up as `VOID`. Start a new capture instead.

### Parallel capture

Capture is the slowest stage. `--capture-shards N` (default 1, at most 8) splits the capture jobs over N engine shards that run
at the same time, on ports `--engine-port` to `--engine-port + N - 1`.
N is clamped to the number of CPUs the process may use, so asking for more shards than cores runs fewer. The comprehension
phase gets faster roughly in proportion to the shards; the later phase that redoes jobs still runs one shard at a time.

- **One parser.** All shards share a single managed parser. Only one parser process is started, however many engines run.
- **Per-job id reset.** Before every capture job the driver resets the engine's id counters. A job's output therefore does not
  depend on which jobs the engine ran before it, which is what lets the jobs be split over shards and then merged.
- **Merge.** Each shard writes a report. The existing gated merge counts only complete reports with a clean teardown, and refuses
  the capture unless they all share one shard count and cover every shard exactly once.
- **Fail-closed.** A shard that stops early or crashes leaves no complete report, so the merge refuses the capture. Nothing is
  filled in around the gap. On a failure or SIGTERM the driver stops the shard children and the shared parser.
- **Resume.** A capture keeps its shard count for life. A resume reads the count from the shard reports that already exist, and
  `--capture-shards` does not change it.
- **Logs.** Each shard child writes `shard-<i>-of-<k>.log` and `shard-<i>-of-<k>.result.json` in the capture directory.

### Warm capture slot

Booting the capture's processes (a parser start and a grammar image load, for the syntax step and again for the run)
costs more than the capture itself on a small append. `mk slot serve` keeps one armed engine and the managed parser up in a
long-lived process, and a capture whose config names its socket (`capture_slot`) takes a lease on them instead: one lease at
a time, held over a unix socket through the syntax step, the run and the merge, and released when the connection closes, so a
capture that dies releases it too. The slot grants a warm lease only when its engine, parser and code pins equal what the
capture seals, the engine and parser are the processes it booted, the readback is clean and the engine is idle; otherwise it
recycles, or frees the ports and grants a cold lease under which the capture runs as before. A warm capture runs one shard on
the slot's engine, re-checks the leased processes in `/proc` instead of launching, and ends each attempt with a release
receipt that the merge gate accepts in place of a teardown. The engine is reused only after a release that ended `complete`.
With no slot answering the capture runs cold, unchanged; with `capture_slot` unset nothing starts a slot.

## 3. Serve (`mk ask`)

`serve.answer.answer(question, cfg)` opens a `Server` over a conversation root and a Neo4j group:

1. **Frame.** The question's shape comes from the wh-word rules alone: no Jev call, no frame cache, no query shard.
   `/context` `frame` carries `shape` only.
2. **Retrieval.** A sealed recipe (`search.kg_recipe.SEALED_RECIPE`) searches the group's key index; the query vector is cached
   in a process-wide LRU keyed by the bonfire's `query_cache_key` and the question's sha256.
3. **Context.** `search.kg_context` assembles a bounded evidence context from the hits.
4. **Answer.** One chat call answers from that context only, with a short-answer prompt and a per-question directive.

Neo4j is opened read-only. Without `--allow-paid`, any step that would need a provider call (the query embedding and the
answer call are the only ones left) and has no cached result raises a refusal and the miss is reported; nothing is sent.

## 4. Provenance: which code made this

A build records the code that produced it, so that a later reader can check that frozen modules did not change.

- **Code identity is the imported package.** `provenance.package_dir()` is the directory of the `memory_kernel` package that is
  running: a source checkout, a wheel install or a copy. Pinned files (`src/memory_kernel/<rel>` keys) are hashed there. The data
  root (`--root`) is only the base of data paths and never of code. `kernel_head` is the git head of the checkout around the
  package, or `unknown` for an installed wheel.
- **The census.** The driver writes `logs/mk/code-census-<sha12>.json` once, at its start: the hashes of the package's files.
  Nothing compares it afterwards. With `--freeze-manifest`, every stage from `graph_picks` through `ctx` runs the provenance
  check of the package against that manifest (`provenance.check`) before its work; `select`, `boundary`,
  `capture` and the stages from `build_dry` on do not run it. The preflight only checks that the manifest file exists.
- **The extraction model.** The capture manifest and the syntax manifest carry `extraction_model` (`{name, version}`), and the
  `driver-start` event's `recipe` of every run carries the same. They are records, not pins: a capture resume and an append
  refuse when the current model's name differs (`MK_EXTRACTION_SPACY_MODEL=<recorded>` or `--allow-parser-change`); no
  record means a build from before 1.4, parsed with `en_core_web_sm`.
- **The ledger shard** pins `ledger.MK_CODE`, the code that built it.
- **`--harness-root`** names a directory of harness files and is needed only to open a shard that a harness (outside this package)
  made, because such a shard pins files outside the package. A build made by `mk` never needs it.
- **Reading the source.** Some docstrings name modules of the reference harness that this code was ported from; they resolve
  to nothing in this repository.

## 5. Runtime data (`mk setup`)

Everything a build reads that is not Python code is fetched by `mk setup`, pinned by sha256 in `data/runtime-pins.json`, verified
in a staging directory and only then moved into `<data>/` (`--data-dir`, else `$MK_DATA_DIR`, else the user data dir).
`<data>/setup.json` is the receipt. `mk setup --check` reads only the receipt and the placed objects.

PropBank frames, the spaCy model and WordNet are fetched from fixed URLs: a commit, a release, and the
`nltk_data` commit that last changed the WordNet file. The FCG Go zip and its grammar image have no versioned download URL
upstream (only the current file is served), so their sha256 pins are the only guard: a changed upstream file is refused, and
`--allow-unpinned-engine` is the only way to place one.

| component | what | placed in |
|---|---|---|
| `fcg-go` | the FCG Go engine binary, fetched through `pyfcg` | `fcg-go/<sha12>/` |
| `fcg-image` | the grammar image the engine loads | `fcg-go/resources/` |
| `propbank-frames` | upstream PropBank frames at one commit | `propbank-frames/<commit12>/` |
| `spacy-model` | `en_core_web_sm` 3.8.0 | the running interpreter's site-packages |
| `spacy-model-trf` | `en_core_web_trf` 3.8.0 (needs `memory-kernel[parser]`); the default for the turn parser (`MK_TURN_SPACY_MODEL`) and for extraction (`MK_EXTRACTION_SPACY_MODEL`, sm opts out) | the running interpreter's site-packages |
| `wordnet` | the NLTK WordNet corpus | `nltk_data/corpora/` |
| `tiktoken` | the `cl100k_base` BPE file | `tiktoken/` |
| `parser-model` | `benepar_en3`, the parser's constituency model | `nltk_data/models/` |

The build reads the engine, image and frames from the data dir by default (`--engine-binary`, `--engine-image`, `--frame-dir`
override). A binary or image whose hash differs from the pin is refused unless `--allow-unpinned-engine` is given; the build
then records `pinned: false` and the actual hashes. Packaged annex frames (24 relational-noun framesets) come with the wheel
and are searched after the upstream frames.

## 6. The parser service

The capture stage needs a syntactic parser: for each message the engine asks a parser host for a constituency and dependency
analysis. `memory_kernel.parser_service` is that host. `pipeline.engine.ParserService` starts it as a subprocess on a loopback
port (`--parser-port`, default 9801, range 9800-9899; refused when busy), waits for `/health`, and stops it when the attempt ends.
The service serves two routes (`/spacy-api/syntactic-parser` and `/syntactic-parser`) and `/health`, using spaCy with benepar
attached. It needs the `parser` extra and the `parser-model` component.

Message text therefore stays on the machine by default. `--parser-url` uses an external parser instead and `--allow-remote-parser`
is required when that parser is not local. The managed parser's URL is sealed into each capture, which is why a capture can only be
resumed on the port it started with.

## 7. Where the paid providers are called

Two external services are used. OpenRouter serves the Jev decision requests of the paid stages (the build sends no chat completions) and the chat completions of
`mk ask`. A request is canonicalised and hashed, and the ledger shard keeps the request and the response, which is what makes a
resume free. Voyage AI embeds:

- the conversation's messages in `events`, which embeds them to cut the conversation into segments, through the stage's own
  cache (`logs/10-events-voyage-cache.json`): only a message that is not in it is sent, and without a key such a message stops
  the stage. `--embedding-cache` does not serve this stage;
- the graph's search keys (`write`, `rewrite`, `index`) through
  `pipeline.embed.embedder`. For these three stages `--embedding-cache` supplies vectors: a cached text is never sent, and a
  text the cache lacks goes to Voyage only when a key is set (without one the stage stops);
- the query in `mk ask`, which reads its cached query vectors and calls Voyage only with `--allow-paid`.

All of these carry conversation text: `events` sends the messages themselves. The models are constants next to the senders that use them, not configuration.

## 8. The runtime grammar (1.4, unreleased)

### Two grammars

The kernel has two parsing layers with different jobs.

- **The extraction parse (§2).** It runs at build and append time, and is slow (a constituency parser, the extraction
  grammar's FCG engine, language-model passes). It decides what goes into memory.
- **The runtime grammar.** It runs on every incoming turn, in milliseconds, with no language model. It is a separate
  FCG grammar of *effectful* constructions on its own engine. Its effect is what the agent does next: which memory
  operations run, which tools the turn may use, and what answer is expected.

A turn the runtime grammar does not read takes today's path unchanged: `/context` and all the agent's tools.

```text
turn ── normalise (spaCy: lower-cased, verbs as marked lemmas) ── runtime FCG engine ── reading (constructions applied)
                                                                                           │
                                     gate (executable) ── refused ──> repair (diagnostics, recall; licence per R1)
                                                           │
                                                      program ── execute: value and memory keywords (read-only graph)
                                                                     │
                                         tool step reached ──> needs_tools (the agent runtime runs it, then resume)
                                         end ──> answered / repair (empty result)
```

### The parts (`memory_kernel/runtime/`)

| module | role |
|---|---|
| `program.py` | the program model (meaning predicates → ops over variables), `OpSpec` / `ToolSpec`, the `Registry` and `validate` |
| `executor.py` | runs value and memory keywords in dependency order, stops at the first tool step whose inputs are bound, replays on resume (`prior`, `resumed`); `procedure` / `seq` / `distribute` order a taught body's tool steps |
| `ops_memory.py`, `lexicon.py` | the memory keywords (read-only Cypher, scoped by the bonfire's group) and the lexicon keywords (`frames-of`, `wh-role`, `role-kinds`: the bonfire's rolesets over the shipped question map) |
| `keywords.py` | the keyword glossary, generated from the three spec maps ([keywords.md](keywords.md), `mk keywords`) |
| `normalise.py`, `engine.py` | the normaliser (with a token map back to the surface text) and the runtime FCG engine: its own process and port, the pinned binary, shuffle off and id counters reset before every reading, so a reading is deterministic |
| `constructions.py`, `base.py`, `store.py` | construction specs and their compiler, the shipped base set (`data/runtime-grammar/en/base.json`), and the bonfire's construction store (one `TaughtPhrase` node per live row, with its standing as counters), and `graph_store.py`, the narrow writer, loaded in ascending precedence |
| `teach.py` | teaching: drafts (admission: the self-read, the duplicate check, `competes`), confirm |
| `api.py` | `Runtime`: `plan`, `resume`, `teach`, `confirm`, the plan cache, the turn window, the gate and the hand |

The keywords are a closed set: a construction composes them and never defines one. The set changes only in a kernel
release, and mining and teaching never add one, because `validate` refuses an unknown op. Tools are declared by the
agent runtime (`ToolSpec`: args, `side_effect`, and the state cells it `reads` and `writes`). **The kernel never runs a
tool**: the executor suspends at a tool step and returns it in the plan.

### The contract

`Runtime.plan(turn) -> TurnPlan` and `Runtime.resume(plan_id, results) -> TurnPlan`. The fields, statuses and gate are
in [runtime-contract.md](runtime-contract.md), the single authority (the README links it). In short, a reading becomes a
program only when all of these hold:

- it is complete;
- a root applied;
- the do-support guard passes;
- the program validates;
- no output is left unapplied, so a constraint no keyword applies yet (a noun restriction, a possessor, a time anchor)
  is a repair, never an answer that ignores it;
- no noun phrase is quantified.

A program that runs and finds nothing is also a repair: a name that names nobody, a verb that names no frame, or an
empty answer. A repair is never a guessed answer: it carries diagnostics, the fallback `recall` when the result was
empty, and the licence of rule R1 (a program that names any tool carries its own licence on every outcome; `licence:
null` only on an unread turn or a repair of a program that names no tool), and the agent runtime falls back to today's
path.

How a plan is kept and resumed:

- Plans live in the `Runtime`: the latest 256, each for 15 minutes after its last use.
- A chat's turns are planned and resumed one at a time, so the turn window ("it", "that one") always sees every earlier
  turn of the chat.
- A resume never re-runs a completed memory keyword.
- The `hand` lists the constructions the turn drew. Their standing (uses that answered, uses that failed, uses the
  room caused, confirmations) is recorded once, when the turn ends. A repair whose verdict is `precondition_failed`,
  `declined` or `tool_error` is a world use: it counts `uses_world` and never lowers the construction's precedence;
  every other repair (a misreading, an empty result, an unresolved name) is a failed use.

### Teaching

`teach(TeachRequest) -> Draft` compiles a definition with no language model. Each form becomes a root construction.
Each body clause is read by the live grammar and becomes one step of the procedure. Teaching is admission (1.5.1):
the draft reads only its own renderings, before it loads (what the live grammar reads them as: `competes`, as frames)
and with it loaded on top (the self-read: each must read through its form). A form with a live construction's
skeleton is refused unless `replace` is set (never over the base set or a catalogue card), and a duplicate (every form
already live with the same meaning up to slot names) is refused naming it. A broad form is admitted: competition is
settled at read time, where both readings reach the agent as frames.

A draft is staged: it is loaded below everything live, and only its teacher can use it, only in the chat where it was
taught. `confirm(draft_id, by)` is for the teacher only. It reads nothing, makes the draft live bonfire-wide, and
records any replaced rows. A teach costs its body's and its own renderings' reads (under 2 s). Both hold the engine
lock.

### Serving (`serve/runtime.py`)

`mk serve` holds one `Runtime` per bonfire (`Runtimes`). Its routes are listed in the README. How the engines are
managed:

- A bonfire's engine starts on its first plan or teach.
- Ports come from `MK_SERVE_RUNTIME_PORTS` (default `9712-9719`; 9711 is `mk plan`'s), and at most
  `MK_SERVE_MAX_RUNTIME_ENGINES` engines run (default 2: the host limit of two FCG engines). A port something else
  listens on (`PortBusy`) is skipped for the next free one. An engine's `engine.json` names its owner process, and an
  engine is reclaimed as an orphan only once that owner is gone.
- Past the cap, the least recently used idle bonfire's engine is stopped by pid. Its `Runtime` is kept (a resume never
  needs the engine), and its engine restarts on its next plan.
- While a bonfire teaches or confirms, its plans wait up to `MK_SERVE_TEACH_WAIT_MS` and then answer with an `unread`
  plan ("runtime busy: teaching"), and its resumes wait up to `MK_SERVE_RESUME_WAIT_S` before a `503` with
  `Retry-After`.
- Every engine stops when serving stops.
- The runtime reads the bonfire's group through READ sessions, and its `recall` is the
  bonfire's `/context` from the caches only (`Pool.context(paid=False)`): a plan never makes a paid call, and a cache
  miss is the diagnostic `recall unavailable: no cached context`.

- Taught phrases, their standing counters and the catalogue are written through the `NarrowWriter` (a sync Neo4j
  driver): its statements touch `TaughtPhrase` nodes and `BonfireMeta.catalogue_json` only, and counters are added to,
  never set. Staged drafts stay in the process's memory.
- A runtime reads its rolesets (`Roleset` nodes) and stored rows once, when it is made.

`mk plan BONFIRE_ID "text"` runs the same `Runtime` once, for debugging. It opens the store read-only
(`ConstructionStore(read_only=True)`) and records no standing (`Runtime(record_standing=False)`).
