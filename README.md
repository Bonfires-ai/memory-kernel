# memory-kernel

Build a typed memory graph from a conversation, then ask it questions.

- `mk build --seed` reads a conversation (a JSON file of dated sessions) and seeds a **bonfire**: a typed graph of events,
  participants and their relations in **Neo4j**. It runs a fixed sequence of stages and keeps every stage's output on
  disk, so a build can be stopped, inspected and resumed.
- `mk append` adds one closed session to a bonfire. Only that session's messages cost.
- `mk ask` takes a bonfire and a question, retrieves the evidence from the graph, and returns the retrieved context and a
  short answer.
- `mk serve` serves every bonfire's context over HTTP and, with the [runtime grammar](#runtime-grammar), reads each
  incoming turn into a program of memory operations and tool steps (`mk plan` does it for one turn).
- `mk inspect` reports the per-stage status of a build, read-only.
- `mk setup` fetches and verifies the runtime data a build needs (the grammar engine, PropBank frames, models, corpora).

The retrieval is planned by a construction grammar (the wh-word rules that read the question's frame) over the graph's
Saga boundaries (walk or segmenter), picks and eventuality, rather than by embedding similarity alone. [docs/architecture.md](docs/architecture.md) describes the stages and how they fit together.

## Quickstart

On Linux x86_64 with Python 3.12 and Docker, with an OpenRouter key and a Voyage AI key. This builds the shipped sample,
an invented conversation of 24 messages, and asks it a question. The build takes about five minutes on a 16-thread
machine and costs about one cent.

```bash
git clone https://github.com/Bonfires-ai/memory-kernel && cd memory-kernel
python -m venv .venv && . .venv/bin/activate
pip install ".[build,parser,serve,ops]"
mk setup

export NEO4J_PASSWORD='choose-a-password'
export MK_OPS_URI='mongodb://localhost:27017'
export OPENROUTER_API_KEY=...
export VOYAGE_API_KEY=...
docker compose -f examples/docker-compose.yml up -d --wait

mk build examples/synthetic_dialogue.json --workdir work --group example-dialogue \
    --purpose-file examples/purpose.txt --seed
mk ask example-dialogue "What is Tomas building from a kit?" --allow-paid
```

The answer is "A telescope". [examples/README.md](examples/README.md) walks through the same steps one at a time.

## Requirements

- Linux x86_64 and Python 3.12 or newer. `mk setup` pins binaries for that platform only.
- A Neo4j 5 server for the graph and a MongoDB server for each bonfire's spend and append lease
  ([examples/docker-compose.yml](examples/docker-compose.yml) starts both).
- API keys for the paid stages (see [Providers and cost](#providers-and-cost)). Only `select` needs no key, no runtime data and no
  Neo4j: every later stage, up to the dry run, can send paid requests. A missing OpenRouter key stops the build at the first
  stage that sends one (`boundary`). A build without a Voyage key is refused at preflight, before any stage runs, when
  a stage that embeds text is pending without the cache it reads: `events` without its own cache file, and `write`,
  `rewrite` or `index` without `--embedding-cache`. `sanity` writes a receipt and embeds nothing.
- Disk space for the runtime data and the Python packages (torch, for the `parser` extra, is large).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install ".[build,parser,serve,ops]"
```

| extra | adds | needed for |
|---|---|---|
| *(base)* | `httpx`, `neo4j`, `nltk`, `numpy`, `rank-bm25`, `spacy`, `tiktoken`, `platformdirs` | `mk ask`, `mk inspect`, `mk setup --check` and the `--adopt-*` options, the `select` stage |
| `build` | `pyfcg` | the grammar engine of a full build, `mk setup` fetching the engine and its grammar image, and the runtime grammar's engine (`mk plan`, the runtime routes of `mk serve`) |
| `parser` | `benepar`, `torch`, `transformers`, `fastapi`, `uvicorn`, ... | the local syntactic parser that the `capture` stage starts |
| `serve` | `fastapi`, `uvicorn` | `mk serve` |
| `ops` | `pymongo` | each bonfire's `kernel_ops` record: `mk build --seed`, `mk append`, `mk bonfire`, `mk serve --ops-uri` |
| `dev` | `pytest`, `ruff` | working on the package |

Which extras each command needs:

| command | extras |
|---|---|
| `mk serve` (context, and plain runtime routes without an engine) | `serve`, plus `ops` with `--ops-uri` |
| `mk build --seed`, `mk append` (extraction runs on `en_core_web_trf` by default) | `build`, `parser`, `ops` |
| `mk build` without `--seed` | `build`, `parser` |
| the runtime grammar's engine (`mk plan`, the runtime routes of `mk serve`) | `build` (pyfcg), plus `mk setup --component fcg-go` |
| GPU parsing (opt-in, `MK_SPACY_GPU=1`) | `gpu` |

`mk setup` installs the 436 MB `en_core_web_trf` pipeline only when the `parser` extra is installed. Without it, `mk setup`
skips trf with a note and `mk setup --check` reports it OK with that note; serve and query fall back to `en_core_web_sm`,
and a build needs `parser` (or the opt-out `MK_EXTRACTION_SPACY_MODEL=en_core_web_sm`).

The spaCy pipelines are wheel URLs, not PyPI names, so they cannot be extras. `mk setup` installs them into the interpreter
that runs `mk`. It uses pip, so in a venv that has none (`uv venv`, for one) it prints the commands to run instead.

## Set up the runtime data

```bash
mk setup                 # fetch every pinned component, verifying each sha256 before it is placed
mk setup --check         # report what is present; no network, no writes (exit 0 = all ok, 1 = something missing)
```

`mk setup` places the data in `--data-dir`, else `$MK_DATA_DIR`, else your user data directory, and records a receipt
(`setup.json`) there. The components are `fcg-go`, `fcg-image`, `propbank-frames`, `spacy-model`, `spacy-model-trf`,
`wordnet`, `tiktoken` and `parser-model`. `spacy-model-trf` (`en_core_web_trf`, 436 MB) is the runtime's turn parser,
`MK_TURN_SPACY_MODEL`, and the default for extraction (build and append), `MK_EXTRACTION_SPACY_MODEL`
(`en_core_web_sm` opts out); it needs `memory-kernel[parser]` (`memory-kernel[gpu]` with `MK_SPACY_GPU=1` adds speed;
on CPU a turn parses in about 90 ms, and a build takes noticeably longer than on sm).
Every one is pinned by a sha256; a mismatch is refused. Only the two engine components
(`fcg-go`, `fcg-image`) accept `--allow-unpinned-engine`, which records `pinned: false` and the actual hashes in the receipt
and in the build's provenance. Behind a proxy or offline, use the `--adopt-*` options to place data you already have
(`mk setup --help`).

Things `mk setup` can leave behind outside the data directory: pip's HTTP cache keeps the spaCy wheel when you install it
with pip.

## Start Neo4j and MongoDB

```bash
export NEO4J_PASSWORD='choose-a-password'
export MK_OPS_URI='mongodb://localhost:27017'
docker compose -f examples/docker-compose.yml up -d --wait
```

`mk` reads the password from the environment variable `NEO4J_PASSWORD` (or from `--env-file`, or, opt-in, from a running
container with `--secrets-from-container`; the environment ranks first). It connects to `bolt://localhost:7687` as `neo4j`
unless you pass `--neo4j-uri` and `--neo4j-user`. The first write needs an empty group: the group name is yours to choose.

MongoDB holds one `kernel_ops` record per bonfire. `mk` reads its address from `MK_OPS_URI` or `--ops-uri`. Prefer the
variable for an address with credentials: a command line is visible to other users of the host.

## Providers and cost

The paid stages send message text to external services. Read this before you point `mk` at private data.

- **Decisions** (the Jev requests of the paid build stages, and the answer call of `mk ask`) go through OpenRouter. Set `OPENROUTER_API_KEY`.
- **Embeddings** go to Voyage AI: the conversation's messages in the `events` stage (it embeds them to cut the
  conversation into segments), the graph's search keys (`write`, `rewrite`, `index`) and the query in `mk ask`. `sanity`
  writes a receipt and embeds nothing. Set `VOYAGE_API_KEY`.
  - `events` keeps its own cache, `logs/10-events-voyage-cache.json` under the conversation root, and sends only the
    messages that are not in it.
  - For `write`, `rewrite` and `index` you can pass `--embedding-cache FILE` (a text-sha to vector JSON): a cached
    text is never sent, and a text the cache lacks goes to Voyage only when the key is set (without it the stage stops). It
    does not serve `events`.
  - `mk ask` calls Voyage for the query only with `--allow-paid`.

  A build is refused at preflight when one of those build stages is pending and it has neither the key nor the cache that
  stage reads. The cache is checked for presence only: a text it lacks still stops the stage when it runs, unless the key
  is set.
- **The models are fixed by the package**, not read from a config file: the build's request builders and the answer step
  name their models in code. To use a different provider or model you change the code.

The driver checks a spend cap before a paid stage: `--cap-usd` (default 20) stops the build
instead of exceeding it, and the paid requests are recorded in `<workdir>/cost-ledger.jsonl`. Provider requests and their
responses are kept in a ledger shard under the conversation root, so resuming a build reads them back instead of paying
again. `--through STAGE` stops after a stage. The default is `build_dry`: every stage before the Neo4j write, which includes
the paid ones. Writing to Neo4j is always explicit: `--seed`, or `--through write` (or `rewrite`, `index`, `sanity`, `seal`).

## Build

The build is 14 stages (`select`, `boundary`, `capture`, `graph_picks`, `graph_a`, `events`, `ctx`, `referents`,
`build_dry`, `write`, `rewrite`, `index`, `sanity`, `seal`).

The shipped example is a short invented dialogue ([examples/README.md](examples/README.md) walks through it):

```bash
mk build examples/synthetic_dialogue.json \
    --workdir work --group example-dialogue --purpose-file examples/purpose.txt --through select
```

`--purpose-file` (or `--purpose`) is optional: the purpose is recorded in `logs/mk/build-config.json` and nothing
reads it; without either, `purpose: null` is recorded. `examples/purpose.txt` holds a general-purpose one.

`--seed` runs the whole pipeline and makes the group a bonfire. `--group` must equal the input's `sample_id`, which
becomes the bonfire's id:

```bash
mk build my_conversation.json --workdir work --group my-conversation \
    --purpose-file examples/purpose.txt --seed
```

A build through `seal` without `--seed` writes the group but no `BonfireMeta`, so `mk ask`, `mk append` and `mk serve` do
not see it as a bonfire.

`--capture-shards N` (up to 8) speeds up the slowest stage, `capture`, on a machine with spare cores. It runs N engines side
by side, behind one shared parser, and is clamped to the CPUs available.

Run it again with the same arguments to resume. The conversation root is `work/<sample_id>`.

### Conversation input

The input is a JSON object `{sample_id, conversation: {session_1: [{speaker, dia_id, text}, ...], session_1_date_time:
"H:MM am|pm on D Month, YYYY", ...}, qa?}`, or a list of such objects together with `--sample-id`. The number of sessions is
not bounded. Three fields are optional; an input without them is read exactly as before.

- `conversation["agent_handles"]`: a list of the handles an agent (a bot) answers to in this conversation.
- Per turn `addressees`: a list of the names the turn speaks to.
- Per turn `timestamp`: an ISO 8601 time with a timezone. It is stored in UTC. A turn stamped earlier than the turn before
  it in the same session is refused, naming the turn's `dia_id`.

**Agent mentions.** A mention of an agent handle (`@handle`, case-insensitive, not inside a longer word or an address)
anywhere in a turn makes that agent an addressee of the turn (an addressee naming a handle, with or without `@` and in
any case, is the same agent). At the start or the end of the text the mention is stripped
before the text is parsed; in the middle it becomes the bare name. A turn that is only a mention becomes empty and is dropped.

**Cleanup.** Before parsing, links become `[link: host]`, data URIs become `[embedded web page]`, HTML becomes its visible
text, and a run of 80 or more characters without a break becomes `[blob]`.

**Splitting.** A long message is split into parts with the ids `<id>#1` to `<id>#n`; a part has at most 1,900 characters and
at most 20 sentences or lines. An image caption belongs to part 1 only. The parts of one message are one arrival for the
episode boundaries: parts 2 to n continue the episode of part 1 with no call. The build's message count is the number of
messages after splitting and dropping (each part counts); for an input without split or dropped turns it is the turn count.

**"You".** When any turn of the conversation carries `addressees` (or a mention made one), "you" resolves only through the
`addressees` of the message it appears in. Otherwise the two-party rule applies: "you" is the other speaker. Messages are
ordered by their order in the source, never by the digits in their ids.

### Limits

The stage caps, the boundary cap and the cap on the round ledger are those of 1.0.0 up to 369 messages; above that they
scale with the message count (messages / 369). A stage cap is recorded, not enforced: a stage whose plan, estimate or
spend goes past it records `stage_cap_exceeded: true` with the cap and the spend or estimate in its plan, summary or
result, logs a `stage-cap-exceeded` event in `driver-events.jsonl`, and goes on. The write stage's Voyage cap ($0.50) and
the driver's per-conversation limit (1.5 x $0.00062 per message; `conv_limit_exceeded` and a `conv-limit-exceeded` event)
are recorded the same way.

The spend cap (`--cap-usd`, default 20) is the one spend stop. It is checked before every paid stage; a Jev
request whose hold would cross it is refused; past their stage caps, the boundary, picks and referents stages refuse
such a hold too and the write stage stops after the Voyage request that crossed it (below them, the check before the
stage holds). The other hard limits are structural: in a build, a shard refuses a hold past its round
ledger's cap or its integrated cap, which a new shard records as $25, so a
conversation of more than 4,612 messages (split parts counted) is refused before any stage, and a conversation has at
most 4,096 messages (a fixed limit of the source step, to be lifted later). An append is bounded by its own budget
instead (see Spend).

- `--boundary {walk,segmenter}` (default `walk`): how episodes are cut. `walk` is the Jev topic walk, unchanged by
  1.3 and the right choice for chats; `segmenter` is for talks and documents: it cuts each session by embedding (a
  target of 60 messages per episode, a ceiling of 150, time gaps of 12 hours kept), sends no Jev request, and needs
  `VOYAGE_API_KEY` (its Voyage spend is a line on the `boundary` stage's cost row, under `--cap-usd`). A `walk` build
  records no `boundary_policy` in `logs/mk/build-config.json`. An append inherits the bonfire's value; on a `segmenter`
  bonfire each appended session becomes new Sagas and the numbering continues.
- `--referents {phrase,structure}` (default `structure`): how entities are keyed. `structure` keys referring cores,
  takes kinds from evidence and names events by frame; `phrase` keys the whole filler, as every bonfire built before
  this flag. `structure` reads WordNet, so it is refused without it (run `mk setup`, or pass `--referents phrase`). A
  `phrase` build records no `referent_policy` in `logs/mk/build-config.json`. An append inherits the bonfire's value.
- `--max-shift-share X` (default 0.10): the largest share of same-day arrivals that may start a new topic.
- `--strict-gates`: the advisory quality gates (the boundary stop rules; in an append also the
  `ctx-prior-fallback` share) are recorded as metrics and stop the build only with this
  flag. An append inherits the bootstrap's config, so **do not build a bonfire's bootstrap with `--strict-gates`**: a
  one-message session (a lone mention, a link) would stop on a share computed over one or two messages.
- `--genesis`: a genesis build, the bootstrap of a bonfire that starts from a few Setup sentences. One check of a full build refuses the zero such a seed
  gives and accepts it only with this flag: graph-A takes zero picks when the recorded picks plan counted no message. The
  flag is recorded as `genesis: true` in `logs/mk/build-config.json`, in the `driver-start` event's `recipe` and in
  `11-seal-001/sealed.json`; a build without it records none of these. A conversation root resumes only with the
  flag it was started with (refused before any stage otherwise). An append of a genesis bonfire is not a genesis build.

A Jev choice that is not the most probable option is corrected to the top option and recorded. A boundary window whose
request would pass the ledger's 31,744-byte limit keeps its most recent messages that fit (the earlier ones of the episode
stay in it, outside the visible window, as in any capacity rollover); a window within the limit is unchanged, and the
boundary summary counts the trimmed steps (`window_trimmed`). The boundary plan stays an untrimmed all-continue estimate.
When not even the prior message and the incoming one fit (a long same-day episode also carries the ids of its earlier
messages), the arrival opens a new episode with no call, as a hard day cut does (`window_capacity_cut`; the plan and
summary count `capacity_cuts`).

To check that the input layer changes nothing for an input that uses none of the optional fields, run
`python scripts/check_input_noop.py <benchmark.json> <old-src-dir>` against an extracted 1.0.0 `src` tree: it prints one
`identical` or `DIFFERS` line per conversation, then the build's message count against the turn count, then whether the
single-conversation boundary plan (the no-send wire list and plan summary, its wires untrimmed and without the 1.3
`stage_cap_exceeded` record) equals 1.0.0's. It covers the prepared batches,
the sources, the message count and the boundary wires, not the graph side.

## Bonfires and append

A conversation that keeps growing (a chat that gets new sessions every day) is seeded once, then extended one session at a
time. That conversation is a bonfire. Its state lives in Neo4j, in the group named by its id, plus one `kernel_ops` record
in MongoDB for its spend and its append lease. No directory survives a seed or an append.

```bash
mk build conversation.json --workdir work --group my-chat --purpose-file examples/purpose.txt --seed
mk append my-chat session.json --scratch-dir work/runs
```

**The seed.** `mk build --seed` runs the full build through `seal`, then writes the bootstrap's rolesets and routes and
the `BonfireMeta` node in one transaction. Serving finds a bonfire only by its `BonfireMeta`, so the group stays invisible
until that last write. `--seed` needs a group equal to the input's `sample_id`. With no `kernel_ops` record, only an
empty group may be seeded. A failed seed is cleaned up when you run it again, and repeating a finished seed does nothing.
`mk bonfire empty GROUP` deletes what a failed seed left behind; it refuses a sealed bonfire unless you pass `--rebuild`.

**`mk append BONFIRE_ID SESSION_JSON --scratch-dir DIR`** appends ONE closed session: a file
`{"conversation_id", "messages": [{"message_id", "speaker", "text", "timestamp", "addressees"?}], "agent_handles"?}` (the
same session `append_session` takes). The run covers that session alone, in a scratch directory that is deleted when it
ends, with the bonfire's config: only the session's messages are parsed, asked about and paid for, so an append costs the
same whether the bonfire holds ten sessions or a thousand. Before anything runs, `mk append` refuses: a bonfire that does
not exist, a held lease, a session whose message ids are all built already (`duplicate`) or only partly (`overlap`), a
session with nothing left after cleanup (`empty`), a chat that is not the bonfire's, or an estimate past the spend ceiling.
**`--on-built skip`** takes a partly built session instead (an edited message that kept its id, for one): its built
message ids are dropped before anything is built or spent, and the rest are appended with their order and timestamps,
the budget counting them alone; the result names the dropped ids (`skipped_ids`). The default, `--on-built refuse`,
refuses it. A `message_id` must be unique across all of the bonfire's chats: under `skip`, an id already built in another
chat is dropped without a warning and recorded in `skipped_ids`.

**One commit per append.** An append reads the bonfire's chain from the graph (entities, name routes, `SAME_AS` links,
per-chat counters and message tails, rolesets) in one read transaction, runs the stages in its scratch directory, and
commits the run in one Neo4j write transaction. Readers see the bonfire before the commit or after it, never between. A
run that stops before `seal` commits nothing: the graph is as it was, the spend is settled, and the lease keeps naming
the session, so the same session must be sent again before any other (`PendingSession`). `mk bonfire release ID` clears
that lease if you abandon the session.

- **Keys are frozen at first mint.** An entity keeps the uuid and key it got the first time its name was mentioned. A
  later session never rewrites an old row.
- **Claims add links and route forward.** When a later session says two names are one person ("Dee is Ana"), the
  append adds a `SAME_AS` edge between the two existing nodes and routes the name to the canonical node from then on;
  it does not merge or delete the old node. The canonical node is the speaker's, when the person is a speaker, else the
  earliest minted. The roster that classifies names is bonfire-wide.
- **"You" is resolved per message, bonfire-wide.** A message is addressed explicitly once some message at or before it,
  in any chat, carried `addressees`; before that the two-party rule applies. A full `mk build` keeps its
  conversation-wide rule.
- **Agent handles** of a session are added to the bonfire's and apply to its messages and every later session's; built
  messages are never parsed again.

**The forward write gate.** An append may only add: it refuses (and commits nothing for) a write that would change or
delete a stored row's frozen fields. A restated fact keeps its first properties. The recipe tag (including
`MK_TEMPORAL_DEIXIS`) must match the bootstrap's. An append's `sanity` writes a receipt (`mk-append-receipt/v1`: gates,
minted, reused, routes, referents) and searches nothing; a full build's is `mk-build-receipt/v1`.

**Referents.** A specific common-noun mention ("the kiln") reuses a stored entity only in two cases: the people talking
share it (same chat, or the speaker or a human addressee took part in it) and it was last mentioned in this chat within
3 days; or Jev judges it the same thing (p(yes) >= 0.5). Otherwise it is a new entity. A new specific referent is
always its own entity (`kiln#1a2b3c4d`); the plain key names the kind. Named mentions, bare (kind) mentions and pronouns
resolve as before. The `referents` stage is paid (Jev yes/no) with a stage cap of $0.05.

**Small sessions.** A session of one to a few messages runs the same stages. The gates that measure a whole
conversation are relaxed for it: a session with no message needing a pronoun pick builds with an empty picks set, the
`ctx` fallback share is advisory (a session with no routed predicate passes `ctx`).

**Search follows `SAME_AS`.** A question that finds one node of a `SAME_AS` component reads the whole component (up to
three hops) as one person: its sheets and timeline are merged and the person is named once, e.g. "Ana (also: Dee)".
A bonfire with no `SAME_AS` edge answers exactly as before.

**Time.** Each session is dated by its own first message (UTC), and the steps and boundary walk of a chat continue from
the chain. A session that arrives earlier than its chat's last stored message is appended in arrival order. The parse,
Jev and embedding work of an append scales with the session, not with the history.

**The budget.** Each append has its own budget (see Spend). When history has to change (append never rewrites it), build
the whole conversation again: `mk bonfire empty GROUP --rebuild`, then seed it again.

**Chats.** A session names its chat in `conversation_id`. The episode boundaries run one walk per chat: a chat's first
message opens an episode with no call, and an episode never spans two chats. Episode ids stay stable when sessions are
appended. In the bootstrap input, tag every session with `session_N_conversation` or none: a partly tagged input is
refused, naming the untagged sessions.

### Spend

Every bonfire's spend is kept on its `kernel_ops` record.

- **An append's budget comes from its own session:** 1.5 x $0.00062 per new message, and never less than $0.10 per
  session. A request whose hold would cross it is refused.
- **The spend ceiling is the operator's brake.** `mk build --seed --spend-ceiling-usd X` sets it at the seed, and
  `mk bonfire ceiling ID X` changes it later. Before any stage, an append holds its estimate against the ceiling: its
  budget, or the last recorded run's budget scaled to this run's new messages, whichever is larger. When the spend plus
  that estimate is past X, the append is refused with "bonfire spend ceiling reached". The estimate is a guess, not a
  bound: a run whose stages plan more than it can still be stopped at X after earlier stages paid. It errs high after a
  short run held at the floor (a 1-message run at $0.10 makes a 50-message session $5), so a tight ceiling can refuse a
  long session. Without a ceiling, nothing is refused on spend.
- **`mk serve` reports it.** `/healthz` lists each bonfire's `spend`, read from `kernel_ops` (`mk serve --ops-uri`):
  `charged_usd` (settled spend plus holds not yet settled) and `ceiling_usd` (null without one). It is `null` without
  `--ops-uri`, and for a bonfire with no `kernel_ops` record.

### Referent policies and kinds

`--referents` is a per-bonfire option stored as `referent_policy` in `logs/mk/build-config.json`. `phrase` keys the
whole filler text and is what every bonfire built before the flag uses (a config with no `referent_policy` reads as
`phrase`, and a `phrase` config is written without the key). `structure` is the default for a new `mk build`. An append
inherits the bonfire's value; an existing bonfire is not re-keyed. `structure` needs WordNet (NLTK data) and is refused
without it.

Under `structure` a filler is cut into referring cores, and each referent has one kind decided over all its mentions:

1. **Person**: the speaker, a roster or deixis person, or a name joined to a person.
2. **A kind noun the text supplies**: a predicative noun ("Brask Labs is a company", said of the PPT role only; an
   adjective predicate, "Kira is rich", supplies nothing), an appositive, or a common noun before a name head ("the
   journalist Milo Dace").
3. **Its own head-noun class**, for a common noun whose head is a NOUN.
4. **Place**, only when it fills a location-type role and its class is location-compatible.

A noun's kind comes from its WordNet lexicographer file: `noun.person` is Person, `noun.group` Organisation,
`noun.artifact` and `noun.communication` Artifact, `noun.location` Place. The first noun sense decides, unless the
first two senses give different kinds. A group noun is an Organisation only when its first sense is
`organization.n.01` or a hyponym of it ("team", "company"); "meeting" or "audience" gets no kind. A capture span that
does not line up with the unit's parse is realigned onto the filler text's token run in the unit (the same characters,
whitespace and tokenisation aside); a filler whose text is not in the unit is parsed on its own, where a head that is
not a noun, pronoun or name refers to nothing. Organisation and Artifact are new node labels; a WordNet Person never
joins a roster. A name inside another core is a mention only, not a separate fact: "the Zorvex spyware" is its own referent
(keyed "zorvex spyware", an Artifact by its head) and Zorvex is mentioned by the message, with no kind from it. A
lowercased name the conversation knows ("the brask labs") is a name. A 3rd-person pronoun possessor is dropped ("their
phone" is "phone"), and a nominal possessor is keyed without its determiners ("the Brask Labs' device" and "Brask Labs'
device" are one entity). A partitive ("this kind of X") refers to X when its
head's first noun sense is `noun.quantity`, or is `kind.n.01` or a WordNet hyponym of it ("sort", "type"); "the
investigation of X" keeps its own referent. A bare noun whose first sense is `noun.quantity` with no complement ("a
lot") refers to nothing.

Entity names are per referent under `structure`, and a message's search key holds its entity names, so the search keys
of a `structure` bonfire differ from a `phrase` build of the same conversation. An Event is named by its frame
("own.01: arg0 Kira and Milo; arg1 Brask Labs"); the sentence is kept as its `source_text`, which answer context shows
on timeline and support lines. The Event's `key_text` does not change.

A `structure` bonfire keeps each entity's kind evidence in its graph. An append decides the kind again over
the stored evidence and replaces a stale kind label on the stored entity. `phrase` labels stay frozen.

### Saga centroids

Under both boundary policies every Saga written carries three properties: `centroid_embedding` (the L2-normalised mean
of its messages' voyage-3 vectors, a list of floats), `centroid_model` (`"voyage-3/document"`) and `centroid_messages`
(the number of messages averaged). They are computed at write from the stage's Voyage cache; the write embeds only the
texts the cache lacks (a few cents at most, as a `voyage` line on the write's cost row) and so needs `VOYAGE_API_KEY` for those. Nothing reads
the centroids yet: they change no answer.

## Alias join

A conversation often names one person two ways ("Dee is Ana", "my name is Ana", "call me Dee"). The build joins those names into one
Person entity instead of two. The rules are cards, in `src/memory_kernel/data/identity-constructions.json`.

What counts as a claim, read from a message's parsed frame:

- an equative "X is Y" where both X and Y are proper nouns in context;
- "my name is X", "I am X";
- "call me X", "name ... X";
- "X and Y are the same person", "X and Y refer to the same person".

Guardrails:

- a first-person claim ("I am X", "my name is X", "call me X") joins X to the speaker only, never to someone else;
- two speakers are never merged, whatever the claims say; such a claim is reported as a conflict;
- a name that two speakers each claim as their own is blocked for both;
- calendar words (month and weekday names, and the like) are never names, nor are possessives ("I'm Bo's");
- a negated ("I'm not X"), irrealis ("if I were X", "I wish I were X") or question ("Am I X?") statement makes no claim.

Later mentions: a name that WordNet does not know as a common noun ("Mary", "Dee") names the joined person in any case and
position ("mary likes mangoes"). A name that is also a common noun ("Rose", "Hope") does so only capitalised and not at the
start of a sentence: "I asked Rose" does; "the rose" and "Rose is lovely." do not. Without WordNet every name is treated as a common
noun. The build and `mk ask` use the same rule.

What the graph shows: one Person entity, whose `name` is the canonical display name (the speaker's own name when a speaker is
in the group) and whose `identity_aliases` property lists the other surfaces, sorted. Nodes without a join have no such
property. The claim sentence itself writes no fact: "Dee is Ana" does not become an `IS` fact between the two names.

Retrieval: `mk ask` reads the joined people of the group once per question (one extra Neo4j read). A question that names a
person by any of their aliases ("What fruit does Dee like?") puts that person's entity sheet first, even when none of
the top messages mention the name. A group with no joined person is served exactly as before.

To see what a build found, run `python scripts/identity_joins.py <path to graph.json>`: it prints the claims, the joins
(alias to canonical) and the conflicts as JSON lines, and writes nothing.

## Ask

```bash
mk ask example-dialogue "What is Tomas building from a kit?" --neo4j-uri bolt://localhost:7687 --allow-paid
```

The first argument is the bonfire's id: its Neo4j group, which must hold a `BonfireMeta` node (a group without one is
not a bonfire, and `mk ask` refuses it with exit 2). Without `--allow-paid`, `mk ask` serves from its in-process cache
only and refuses (exit 2) when a question needs a new provider call. With `--json` it prints the query frame (its
`shape`, from the wh-word rules), the retrieved context and the answer. Nothing is written to disk and the graph is
only ever read.

## Library API

A service can grow a bonfire and ask it without knowing how a bonfire is built:

```python
from memory_kernel.api import SessionInput, append_session, context
from memory_kernel.ops.mongo import MongoKernelOps

ops = MongoKernelOps.from_uri("mongodb://localhost:27017")
session = SessionInput(
    conversation_id="chat-7",
    messages=[
        {"message_id": "m-101", "speaker": "Ivo", "text": "The kiln is warm again.", "timestamp": "2026-06-04T16:05:30+02:00"},
        {"message_id": "m-102", "speaker": "Rua", "text": "Bowls go in tonight.", "timestamp": "2026-06-04T16:06:10+02:00",
         "addressees": ["Ivo"]},
    ],
    agent_handles=["@hearth"],  # optional: added to the conversation's
)
out = append_session("my-chat", session, ops=ops, scratch_root="work/runs")  # {"status", "run", "sessions", "new_messages", "skipped_ids", "result"}
ctx = context("my-chat", "When do the bowls go in?", serve_cfg_overrides={"allow_paid": True})  # {"frame", "text", "message_ids", "tokens", ..., "engine", "run"}
```

**`append_session(bonfire_id, session, *, ops, scratch_root, owner=None, lease_ttl_s=3600, on_built="refuse", overrides=None)`**
takes one newly closed session and blocks for the whole run. `ops` holds the bonfire's lease and spend, `scratch_root` is
where the run is built, and `overrides` are build config fields set over the bonfire's own: the secrets, the Neo4j
address, the host's install paths.

- Sent twice (a retried job): every message id is in the bonfire already, so it returns `status: "duplicate"` and runs nothing.
- A session that shares only some message ids with the bonfire is refused; it is never merged. With `on_built="skip"`
  (`mk append --on-built skip`) its built ids are dropped and only its new messages are appended; `skipped_ids` names the
  dropped ids (empty otherwise). A session whose ids are all built is a `duplicate` either way.
- A session whose messages are all blank or markup only returns `status: "empty"`: nothing runs.
- `status: "appended"` means the run sealed and its commit landed; `"stopped"` means it ran but stopped before `seal`:
  nothing is committed and the spend is settled. **Send a stopped session again, with the same `on_built`, before any
  later session of that bonfire**: sessions are ordered as they reach the graph, and the lease refuses any other session
  with `PendingSession` until this one lands.
- **One job per bonfire at a time.** A second `append_session` of a bonfire while one runs raises `AppendBusy`, which is
  retriable; queue the jobs of one bonfire. The lease is renewed every third of `lease_ttl_s`, and expires by itself if the
  host dies.
- **Agent handles** of a session are added to the bonfire's and every later session's messages are read with them.
- Other refusals raise `BonfireError` (a `ValueError`): no such bonfire, a malformed session (a `timestamp` that is not an
  ISO string with a timezone, `addressees` that are not a list of strings), the spend ceiling, and every refusal of
  `mk append`.

**`context(bonfire_id, question, *, serve_cfg_overrides=None, speaker=None)`** answers from the bonfire's graph in
Neo4j (the Neo4j target, `neo4j_uri` and `neo4j_user`, goes in the overrides) and returns the context block plus
`engine: "kernel"` and `run` (`"bootstrap"` for committed run 0, else the run number). A group without a `BonfireMeta`
node is not a bonfire (`BonfireError`). The query frame comes from the wh-word rules alone (no Jev call), so it cannot
fail. The call is cache-only unless the overrides set `allow_paid`; secrets come from the process environment.
`acontext` is the same as a coroutine, for a caller inside an event loop.

## `mk serve`

```bash
pip install ".[serve,ops]"
mk serve --neo4j-uri bolt://localhost:7687 --host 127.0.0.1 --port 8740 [--ops-uri mongodb://...]
```

`mk serve` answers the context of every bonfire in one Neo4j over HTTP. A bonfire is a group with a `BonfireMeta`
node; its id is the group name. A group without one (still seeding, or a 1.8 group) is not a bonfire. There is no
bonfire directory: nothing is read from or written to disk.

| request | answer |
|---|---|
| `POST /v1/bonfires/{id}/context` `{"query": "..."}` | `200 {engine: "kernel", context, message_ids, tokens, frame, run, degraded: false}` |
| `POST /v1/bonfires/{id}/entities/sheet` `{"name": "..."}` or `{"uuid": "..."}` | `200 {entities: [{uuid, name, kinds, summary, lines: [{text, message_ids, episode_ids}], aliases, identities, timeline}]}`, at most 3, read-only; `404` (empty list) for an unknown bonfire or no match, `503` when the lookup fails, `422` unless the body is exactly one non-empty string |
| the body also carries `"answer_shape"` (a `TurnPlan`'s) | the frame's shape is that shape instead of the wh-word rule's |
| any other error while answering (Neo4j down, a run that does not open) | `200` with `degraded: true`, `context: ""` and a `reason` |
| an unknown bonfire | `404 {engine, degraded: true, reason}` |
| Neo4j does not answer the bonfire lookup | `503 {engine, degraded: true, reason}` |
| `GET /healthz` | `200 {status: "ok", version, bonfires: {id: state}, spend: {id: spend}}` once `mk serve` can answer, else `503 {status: "warming", ...}` (see Warm-up); `version` is the installed package's; a state is the loaded run, `"warming"` or `"failed: <reason>"`; a spend is `{charged_usd, ceiling_usd}` (see Spend; null without `--ops-uri` or a `kernel_ops` record); plus `parser_notes: {query: "sm-fallback"}` while the query parser has fallen back to sm |

**`mk serve` has no authentication.** Every route is open to whoever can reach the port, including `/teach` and the
catalogue `PUT`, and a question can spend on your provider keys. Keep it on `127.0.0.1` or a private network, and put your
own gateway in front of anything public.

**Warm-up.** `mk serve` is ready when it can answer. At start, in the background, it loads its models (the query and
turn parsers, the WordNet corpus, the token encoder) once, then opens every bonfire at its committed run. `/healthz`
answers a 503 until every bonfire is open or has failed and at least one is open, then a 200. A bonfire whose open
cannot reach Neo4j yet (right after a restart Neo4j may come up after the kernel) stays `warming` and is tried again
after 5 s, doubling up to 30 s. A bonfire whose open is refused is listed as `failed: <reason>` and does not hold the
others. A bonfire that appears later is opened when `/healthz` first lists it; it is listed `warming` meanwhile, and a server
that has been ready stays a 200.

**Warm swap.** Each bonfire keeps one server open (its Neo4j driver and loaded index) between questions, with the
`committed_run` of the `BonfireMeta` it was opened at. On each question the `BonfireMeta` is read fresh; a higher
`committed_run` is opened in the background (questions keep the loaded server meanwhile), swapped in, and the old one
is closed when its last question finishes. An append in progress is invisible: its `committed_run` moves in the
append's one commit transaction. A run that fails to open is logged and not tried again for `--open-backoff`
seconds (60); the loaded one keeps answering.

**Paid calls.** `mk serve` sends on a cache miss by default (`--no-allow-paid` serves from the caches only). The query
frame comes from the wh-word rules (no Jev call, no frame cache; `frame` carries `shape` only). The query embeddings
live in a process-wide LRU of 4,096 entries keyed by the bonfire's `query_cache_key` (the first 12 hex characters of the
sha256 of its build config, on `BonfireMeta`) and the question's sha256, shared by every run of the bonfire, so a swap
keeps their hits; a restart starts it empty, and a cache-only server (`--no-allow-paid`, or the library `context()`,
cache-only by default) answers `degraded` ("no cached context") until a paid call warms it. Secrets are resolved as for
`mk ask` (process environment, then `--secrets-from-container`, then `--env-file`) and never logged; the Neo4j target is
`--neo4j-uri` and `--neo4j-user`, not the bonfire's.

**Runtime grammar.** The plan, resume, teach and confirm routes are described under
[Runtime grammar](#over-http-mk-serve).

## Runtime grammar

The kernel parses language twice, with two grammars that do different jobs:

- **The extraction parse** runs at build and append time (PropBank frames, constituency and dependency parses, FCG
  capture). It is slow, and it decides what goes into memory.
- **The runtime grammar** runs on every incoming turn. It is a small FCG grammar of *effectful* constructions on an
  engine of its own, and it reads a turn in milliseconds with no LLM call. A construction's meaning is a **program**:
  memory operations the kernel runs against the bonfire's graph, and tool steps it hands to the agent runtime.

"where does Kira Dace work?" reads as `frames-of(work) → find-entity("Kira Dace") → events(work.01, Kira Dace, arg0)
→ wh-role(where) → role-filler → answer`, and its answer is Zorvex. The grammar is the shipped base set
(`memory_kernel/data/runtime-grammar/en/base.json`: clause types, wh roots, noun phrases, deixis, time) plus the bonfire's stored
constructions (the bonfire's `TaughtPhrase` nodes: taught, mined and hand rows). Verbs resolve through the
bonfire's rolesets and the shipped question map (which role of a frame answers who, what, where, when). The operations
a program may use are a closed set of **keywords**, listed with their rules text in [docs/keywords.md](docs/keywords.md)
(`mk keywords` prints it). Constructions only compose keywords: the set changes only in a kernel release, and neither
teaching nor mining ever adds one.

### The contract: `plan` and `resume`

`plan(turn) -> TurnPlan` reads one turn, `{text, speaker, addressee, participants, chat_id, message_id, at}`, and returns
a plan whose `status` is `answered`, `needs_tools`, `repair` or `unread`. A `needs_tools` plan names the pending tool
step; the agent runtime runs it against its world and reports a verdict row with `resume`. **The kernel never runs a
tool.** A tool is declared in the bonfire's catalogue: its arguments, its side effect, and the state cells it reads and
writes. With `x-returns` it also says which argument its result stands for, so a taught body can refer to that result
("it", "them"). A repair is never a guessed answer: the turn falls back to today's path.

**The licence (rule R1).** A program that names any tool carries its own `licence` on every outcome, repairs included;
`licence` is `null` only on an `unread` turn or a repair of a program that uses no tool.

[docs/runtime-contract.md](docs/runtime-contract.md) is the single authority for the `TurnPlan` fields, the result
rows and verdicts, retry and committed effects, the confirmation policy, slot constraints, standing, `invoke` and the
catalogue format. The gate that decides whether a reading may run is in
[docs/architecture.md](docs/architecture.md#the-contract).

**The turn window.** "it" and "that one" refer to the entity the chat's previous answer named: "what did Milo
publish?" then "who gave it?". A previous turn that named no entity, or more than one, is refused rather than guessed.
Plans are kept for 15 minutes after their last use (the latest 256 per bonfire). A resume of an expired or unknown plan
is an error that says which.

**The hand.** `hand` lists the cards the turn drew: each construction applied, whether base, mined, hand or taught,
its maker, its standing (uses that answered, uses that failed, confirmations), and its meaning as keyword rules text.
Standing is recorded once a turn ends: an answered turn counts a good use and a repair a failed one.

### Teaching

A person teaches a verb in chat. The agent turns the definition into a teach request (an LLM may draft it):

```json
{"forms": ["brew {X} for {R}", "brew {X}"], "body": ["take water", "boil water", "give {X} to {R}"],
 "defaults": {"R": "me"}, "teacher": "Ana", "chat_id": "c1"}
```

Each form becomes a root construction. Each body clause is read by the live grammar, so the body composes what
already reads: catalogue constructions for tools, and earlier taught verbs. A body may not ask a question, refer to the
turn window, or name a quantified noun phrase; it may refer to an earlier step's result ("it", "them"). The result is a
**draft**. It is usable only by its teacher, in the chat where it was taught, until the teacher **confirms** it; it then goes live bonfire-wide.

Teaching is admission: a teach reads its body clauses and its forms' own renderings, and nothing else.

- **A form with the skeleton of a live construction** (the same literals, any slots) is refused unless the teacher
  asks to replace it, and over the base set or a catalogue card always. A replace is recorded, with who replaced what.
- **A duplicate** (every form already live with the same meaning, up to slot names) is refused naming those
  constructions.
- **A broad form is admitted** ("who {X}?" next to the base "who is {X}?"). Where it and another construction both
  read a turn, both reach the agent as competing analyses in the plan's `frames`, and the outcome moves their standing.
- **`competes`**: the draft says what the live grammar reads each form's own uses as now ("look at the office" also
  reads as the look card, place "office"), so the agent can tell the teacher.

A refused teach is a draft with `problems` and changes nothing. A teach takes well under 2 seconds, and a confirm
reads nothing.

### Over HTTP (`mk serve`)

The same `mk serve` adds the runtime routes:

| request | answer |
|---|---|
| `POST /v1/bonfires/{id}/plan` with the turn's fields (`text` and `chat_id` required; `at` defaults to the server's UTC time; optional `committed`: the effects the world already did, `[{tool, args, value}]`; optional `context`: the open frames of the speaker's turn before, at most 8, see [docs/runtime-contract.md](docs/runtime-contract.md#context)) | `200` with the `TurnPlan`, never a `503`: a turn that cannot be read now (the bonfire teaches or reloads its catalogue, every engine is in use, an engine that does not start) is a `200` `unread` plan naming why |
| `POST /v1/bonfires/{id}/plan/{plan_id}/resume` `{"results": [{"step_id", "tool", "args", "verdict", "value", "detail"}]}` | `200` with the `TurnPlan`; `404` unknown or expired plan; `409` a row that does not match its pending step, a duplicate, a resume already running; `503` with `Retry-After` while the bonfire still teaches after the resume waited |
| `POST /v1/bonfires/{id}/plan/{plan_id}/retry` | `200` with the `TurnPlan` (the refused step issued again, same `step_id`); `404` as for resume; `409` a plan with no failed step |
| `GET /v1/bonfires/{id}/catalogue` | `200 {sha256, tools}`; `503` the installed catalogue does not load |
| `PUT /v1/bonfires/{id}/catalogue` (the catalogue file as the body) | `200` as the `GET` after the install; `400` a catalogue that would not load (the previous one keeps serving); `503` busy, retry later |
| `POST /v1/bonfires/{id}/teach` `{"forms", "body", "defaults", "teacher", "chat_id", "replace"}` | `200 {draft_id, specs, problems, competes}` (a refused teach is a `200` with problems; `competes`: form text -> the frames the live grammar reads its own uses as); `409` a refusal; `503` the engine cannot run it now (every engine in use, an engine that does not start, a failure under the teach) |
| `POST /v1/bonfires/{id}/teach/{draft_id}/confirm` `{"by"}` | `200 {draft_id, specs}`; `403` not the draft's teacher; `409` any other refusal; `503` as for teach |
| `GET /v1/bonfires/{id}/constructions` | `200 {constructions: [...]}`: the bonfire's stored rows, live and staged; `503` the bonfire's runtime does not load |
| any of them | `404` unknown bonfire; `422` a malformed body: a wrong type (a non-string speaker, participants that are not a list of strings, ...), a string that is not valid UTF-8, or a field past its cap (text 4,000 characters; ids 256; participants 256; a teach's forms and body 32 entries of 500 characters) |

Each bonfire gets one runtime, and its FCG engine starts on the bonfire's first plan (or teach). At most
`MK_SERVE_MAX_RUNTIME_ENGINES` engines run (default 2), on ports from `MK_SERVE_RUNTIME_PORTS` (default `9712-9719`;
9711 is `mk plan`'s). A port something else listens on is skipped for the next free one. An engine's state file names
the process that started it, and no other process (a second `mk serve`, an `mk plan`) ever kills it while that
process lives. Starting one more engine than the cap first stops the engine of the least recently used idle bonfire,
by pid. Its plans and resumes stay valid, and its engine starts again on its next plan. Engines stop when `mk serve`
stops.

**Teach and confirm hold the bonfire's engine** (a teach well under 2 seconds; a confirm reads nothing). While a
bonfire teaches:

- a plan for it waits up to `MK_SERVE_TEACH_WAIT_MS` (15 s default); past that it answers an `unread` plan whose
  diagnostic is `runtime busy: teaching` (`runtime busy: reloading the tool catalogue` during a catalogue `PUT`), and
  the turn falls back;
- a resume waits up to `MK_SERVE_RESUME_WAIT_S` (30 s), then answers `503` with `Retry-After`.

Other bonfires are not affected.

The runtime reads the bonfire's Neo4j group, read-only, with the address and password `mk serve` resolves for
`/context`. Its fallback `recall` is the bonfire's `/context` from the caches only: a plan never makes a paid call,
and on a cache miss the plan carries `recall unavailable: no cached context` and no `recall` result.
The engine needs the pinned FCG Go (`mk setup --component fcg-go`) and pyfcg (the `build` extra). A bonfire's tool
catalogue is stored on its `BonfireMeta`, installed with the `PUT` route or `mk catalogue install BONFIRE_ID FILE`; recovery after
a plan expires or the catalogue reloads is in [docs/runtime-contract.md](docs/runtime-contract.md#recovery).

**Limitations.**

- **A runtime is made once.** A bonfire's runtime reads its rolesets and stored constructions when it is made (its
  first runtime request). An append after that is not seen by the runtime grammar until `mk serve`
  restarts.

### `mk plan` (debugging)

```bash
mk plan my-chat "where does Kira Dace work?" --speaker Ana --chat c1
```

`mk plan` prints the `TurnPlan` JSON of one turn, using the same `Runtime` on its own engine (`MK_RUNTIME_FCG_PORT`,
default 9711). It reads the bonfire's Neo4j group when a Neo4j password resolves (`NEO4J_PASSWORD`, `--env-file`,
`--secrets-from-container`), and an empty graph otherwise (`--empty-graph` forces it). One stderr line says which
reader it used. It never writes the bonfire: it opens the construction store read-only and records no standing.
`--engine-dir` puts the engine's log and state file elsewhere than the data dir. An argument that is not valid
UTF-8 or a missing pyfcg is refused (exit 2) before any engine starts.

## Where a bonfire lives

| store | holds |
|---|---|
| Neo4j, the group named by the bonfire's id | the graph, plus the bonfire's state: `BonfireMeta` (its committed run, config and tool catalogue), `Roleset`, `Route` and `TaughtPhrase` nodes |
| MongoDB, `mk_ops.kernel_ops` | one record per bonfire: its spend, its open holds and its append lease |

Back up both. A host keeps nothing a bonfire needs: `mk serve` reads and writes no bonfire files, and an append's scratch
directory is deleted when the run ends. Hosts that share a bonfire must keep their clocks in sync to well under the lease's
time-to-live. A crashed writer's lease expires by itself after `lease_ttl_s`, one hour by default; a stable `owner` per
worker can take its own lease back at once.

## Inspect

```bash
mk inspect work/example-dialogue
```

prints, as JSON, which stages are done, pending or stopped, what has been spent, and the last stop reason. It never writes.

## Privacy

- **The syntactic parser runs on your machine.** By default `mk build` starts the in-package parser on a loopback port
  (default 9801, `--parser-port`), uses it for the `capture` stage, and stops it. `--parser-url` points at an external
  parser instead; a parser that is not on this machine is refused unless you pass `--allow-remote-parser`, because message
  text would then leave the machine.
- The paid stages send message text to OpenRouter and Voyage AI (see above). No message text leaves the machine otherwise; what
  does is what `mk setup` downloads and, when the tiktoken file has not been set up and the network is open, the tokenizer
  library's own download of its vocabulary file (no message text is sent; `mk setup --component tiktoken` avoids it).
- Secrets are never printed and never written to a build's files; a stage child receives them through its environment.

## Licences

The project is Apache-2.0 ([LICENSE](LICENSE)). Third-party and fetched material is listed in [NOTICE](NOTICE):

- `src/memory_kernel/data/propbank-frames-annex/` is a set of PropBank framesets licensed CC BY-SA 4.0 (its own `LICENSE`
  and `NOTICE`). Upstream PropBank frames are fetched by `mk setup`, not redistributed here.
- The FCG Go engine and its grammar image are fetched through `pyfcg` and are not redistributed.
- The parser model is fetched by `mk setup`.

See [CONTRIBUTING.md](CONTRIBUTING.md) to work on the code and [CHANGELOG.md](CHANGELOG.md) for the release notes.
