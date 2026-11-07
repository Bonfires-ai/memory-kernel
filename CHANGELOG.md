# Changelog

## 2.0.1 (2026-11-07)

- **A paid Jev call no longer stops the build when the provider adds fields to `usage`.** The archive reader required
  `usage` to hold exactly `input_tokens`, `output_tokens` and `cost`, so a response that also carried `is_byok` and
  `cost_details` stopped the build at `boundary` with `provider usage differs`. The three keys are still required and
  their values are checked as before; extra keys are ignored, and the receipt and the ledger record keep only the three.

## 2.0.0 (2026-11-05)

This release carries everything since 1.4.1. The 1.5 to 1.8 lines were never published on their own, so their notes
follow as sections of this one, newest first.

### 2.0: a bonfire's state lives in the graph

- **A bonfire is a Neo4j group, not a directory.** Its state is the graph plus a `BonfireMeta` node and `Roleset`, `Route`
  and `TaughtPhrase` nodes. Spend and the append lease are one `kernel_ops` record per bonfire in MongoDB. No directory
  survives a seed or an append, so a host keeps nothing a bonfire needs.
- **`mk build --seed` makes a bonfire.** It runs the full build through `seal` and writes the `BonfireMeta` last, so a
  group stays invisible until it is finished. `--group` must equal the input's `sample_id`. `--spend-ceiling-usd` sets
  the ceiling. A failed seed is cleaned up when it runs again, and repeating a finished seed does nothing.
- **`mk append BONFIRE_ID SESSION_JSON --scratch-dir DIR`.** An append reads its state from the graph, builds in a scratch
  directory and commits in one Neo4j write transaction. A run that stops before `seal` commits nothing.
- **`mk bonfire` is now `empty`, `release` and `ceiling`.**
- **`mk ask` and `mk serve` take bonfire ids and read Neo4j only.** `mk ask BONFIRE_ID QUESTION`. `mk serve` answers for
  every bonfire in one Neo4j and is ready with zero bonfires.
- **The library takes ids.** `append_session(bonfire_id, session, ops=, scratch_root=)` and `context(bonfire_id, question)`.
- **The `ops` extra (`pymongo`) replaces `store`.** Set `MK_OPS_URI` or pass `--ops-uri`. Without the extra, `mk build --seed`
  and `mk serve --ops-uri` refuse and name it.
- **Removed:** directory bonfires and their `chain/` and capture cache, `mk bonfire init`, `ground`, `boundary`,
  `backfill-lines` and `prune`, `mk store`, and `mk serve --bonfire`. A directory bonfire does not convert: seed it again from its
  conversation.
- **`examples/docker-compose.yml` starts MongoDB too**, and the README has a tested quickstart.

### 1.8

- **The build archives the taxonomy and question layer: nine stages leave, and the search keys do not change.**
  `taxonomy`, `jev1_pilot`, `jev1`, `taxonomy_events`, `taxq`, `labelq`, `jev2`, `remine` and `graph_b` are no longer
  stages. Retrieval reads only the walk boundaries, the picks and the eventuality, so none of them touched a search key:
  on all 10 reference conversations 1.8's keys are byte-identical to 1.7's (5,882 of 5,882), checked by the key gate after
  the projection change and again after the last build change, and a $0 measurement showed fresh `events` and `ctx`
  calls wire-identical to 1.7's. The build costs $2.36 for the 10 reference conversations, down from $8.21 (provider
  receipts). The build is 14 stages: `select`, `boundary`, `capture`, `graph_picks`, `graph_a`, `events`, `ctx`,
  `referents`, `build_dry`, `write`, `rewrite`, `index`, `sanity`, `seal`. 
- The data file `remine-unit-price.json` is removed.
- **A full build's `sanity` is a $0 receipt.** It writes `mk-build-receipt/v1` (the index's message count, recipe tag and
  stamped keys) and searches nothing; an append's receipt (`mk-append-receipt/v1`) was already one. The blind retrieval check that searched the taxonomy
  questions left with the layer; retrieval quality is guarded by the key gate.
- **New rows no longer carry the layer's codings.** A projection writes no Question, Plan or Purpose coding and no
  `ANSWERS`, `ANSWERS_VIA` or `DERIVED_FROM` edge, and an Episodic row carries no `topic_labels` or `topic_p` (absent keys, not empty lists, so a retried
  write is not refused as a change). `Saga.summary` has no "top labels" clause and a new Saga's `inventory_sha` is null; a
  stored value is kept, and a continued walk Saga that holds one still passes the forward gate.
- **`/context` `frame` carries `shape` only; `label` and `label_p` are removed.** The query frame comes from the wh-word
  rules alone: no Jev call, no frame cache, no query shard (`bonfire/query-shard` is no longer made), and a frame file cached
  by 1.7 is never returned. `mk ask --harness-root` has no effect on serving, since `mk ask` opens no ledger.
- **`--purpose` / `--purpose-file` are optional.** They are recorded in `build-config.json` and nothing reads them; without
  either, `purpose: null` is recorded.
- **Removed flags: `--taxonomy-gate`, `--remine-unit-price-budget` and `--lost-wires`** (the last on `mk build` and
  `mk append`). Old files still load: a 1.7 `build-config.json` has its retired keys dropped, a stored `through` that names an
  archived stage maps to `graph_a` (`taxonomy` maps to `select`), and the retired names in a bonfire's `frozen` list are
  skipped, so a 1.7-built bonfire appends under 1.8.
- **The query cache keeps its key where it can.** The directory under `bonfire/query-cache` is the first 12 hex characters
  of the sha256 of `<bootstrap>/taxonomy/labels.json` when that file exists, so an old bonfire's cache stays warm; otherwise
  of `<bootstrap>/logs/mk/build-config.json`. The key follows that file's bytes, and the file is rewritten
  (`runner.write_config`, at each build start; identical settings give identical bytes) when `relocate` moves the
  bonfire's path and when the bootstrap build is re-run with different settings. Each such rewrite makes the cache go cold
  once. A paid caller embeds its questions again (cents). A cache-only caller (`mk serve` without paid sends, or the
  library `context()`, cache-only by default) answers `degraded` ("no cached context") until a paid call warms the new
  cache. Kernel 2.0 replaces this key.
- **Events receipt counts differ from 1.7's on the same conversation; this is not a regression.** The empty inventory no
  longer merges rolesets that shared a remined card predicate, so the occurrence counts in `occurrences.json`, in the
  occurrences block of `summary.json` and in the stage receipt change. No paid call changes and nothing the projection or
  the search reads: a $0 measurement on 10 conversations found no wire or cache-key change in 4,162 `events` calls and
  6,592 `ctx` calls.
- **An append's cap is max(1.5 x `INDEX_USD_PER_MESSAGE` x messages, `APPEND_FLOOR_USD`).** No 1.8 stage records a plan
  estimate (`est_usd`), so there is no plan term. `INDEX_USD_PER_MESSAGE` is $0.00062, down from $0.0025, re-derived from
  the kept stages' receipts; `APPEND_FLOOR_USD` stays $0.10, and the headroom ratio is unchanged from 1.7.
- **Package removals break any importer.** Gone from the package: `json_utils`, `knowledge.taxonomy`,
  `ingest.llm_adapter`, `ingest.nlp_runtime_utils`, `ingest.nlp_taxonomy_*`, `ingest.nlp_topic_ontology_artifacts`,
  `ingest.taxonomy_union`, `graph.g1_cards`, `graph.g1_codings`, `pipeline.llm.GapSender` and the
  `memory_kernel.ingest.NlpTaxonomyMemoryKernelIngestRunner` export, with the stage modules of the nine stages. Names
  also left kept modules: the `BuildConfig` keywords `taxonomy_gate`, `remine_unit_price_budget` and `lost_wires` (the
  retired keys of a stored `build-config.json` are dropped on load, but a constructor call that passes one fails),
  `ServeConfig.query_shard`, `frame_fallback`, `labels_path` and `shard`, `Misses.jev`, `write.check_c2_rows` and
  `c2_kwargs`, and `embed.VoyageCache` and `batches`. Any consumer that imports one of them breaks. `pipeline/ledger.py` is
  code-pinned to the tag and its comment still names `jev.shard_budget`; `shard_budget` now lives in `pipeline/bonfire.py`.
- **A user `--freeze-manifest` that pins archived files fails with `FrozenCodeChanged`.** A pinned file that no longer
  exists is reported as `<key>: missing` at the first freeze-checked stage (`graph_picks`), after `boundary` has run. Remove
  the archived files from the manifest, or drop `--freeze-manifest`.
- **Upgrade order, and no rollback: a bonfire built or appended by 1.8 cannot go back to a 1.7 kernel.** Deploy 1.8 to
  every host that builds, appends or serves these bonfires, with no append in flight, before building or appending under
  1.8. A 1.7 kernel breaks on 1.8's output in three ways: 1.8 run roots lack the frozen-stage links a 1.7 append reads; a
  1.8-built bonfire has no `taxonomy/labels.json`, so 1.7 serving cannot key its query cache (`api.query_cache`) and
  answers `degraded`; and a build without `--purpose` records `purpose: null`, which 1.7's `BuildConfig` refuses.

### 1.7.0rc4

- **`mk bonfire boundary BONFIRE_DIR segmenter`: a walk bonfire switches its appends to the embedding segmenter.** The
  Jev topic walk makes one model call per message in order (a 159-message session spent 506 s there), while the
  segmenter cuts each session on time gaps and topic shifts over the Voyage vectors the events stage embeds anyway,
  with no Jev call. The switch is recorded in `bonfire.json` (`boundary_policy`, read by `BonfireDir.boundary_policy`;
  `run_config` applies it over the bootstrap's policy) and goes one way: any policy but `segmenter` is refused. On
  each chat's next append its open walk episode ends where the session starts (`chain.segmenter_state`, applied by
  `forward_input` to the boundary block and to the chain view the write stage reads), so that session opens new
  Sagas and links from the chat's last one by `NEXT_SAGA`; runs and Sagas already built are kept. A bonfire without
  the switch is unchanged, and a segmenter append still refuses a chat the walk built; a run override of the policy
  that differs from the bonfire's own is refused (`run_config`). The switch is refused while an append runs, and while
  a stopped append that started writing the graph waits for its retry (`pending_written_runs`, from the driver's
  `start` row of the write stage: its session's rows may be in the group under the walk, so only a walk retry matches
  them). A stopped run that never reached the store (a killed store-backed worker) is not seen from another directory:
  switch on the kernel worker's own working copy, after its outbox has drained. With a store (`--uri` or `MK_STORE_URI`) the switch runs
  under the bonfire's lease (`api.store_set_boundary_policy`: hydrate the current version, or keep a copy dirty at
  it, switch, persist the next); without one a store working copy is refused, since the next store-backed append
  would hydrate over a local change. A new bonfire can be built with `--boundary segmenter` instead. Version
  `1.7.0rc4`.
- **Upgrade order: no rollback below 1.7.0rc4 after a switch.** An older kernel ignores `boundary_policy`: it continues
  the walk on chats not yet appended since the switch, and refuses at the boundary stage every chat the segmenter has
  closed. Deploy rc4 to every appender first, then switch.

### 1.7.0rc3

- **"this Friday" names the Friday ahead.** The relative-time note (`content_with_relative_time`, and the
  `happened_at` the G1 projection reads back from `relative_resolution_note`) read a weekday with no `last`/`next` forward
  only for a future-tense sentence or a nearby `next`, so a plan said in the present tense ("we demo the plugin on
  staging this Friday", said on a Thursday) was dated to the Friday just gone. A weekday after a forward modifier
  (`this`, `coming`: the lexicon's new `weekday_forward_modifiers`) now reads forward whatever the tense; the anchor
  day itself is still an allowed answer. A bare weekday keeps its rule, so a past story's "on Friday" still reads
  back. The note is computed when a message is read, so messages already stored read the new date; a `happened_at`
  already written keeps its date. Version `1.7.0rc3`.
- **Upgrade step: persist the search keys again.** The note is part of a message's search key, so a bonfire that
  already holds a "this <weekday>" or "coming <weekday>" message reads `IndexStale` ("N stale ... message keys;
  run persist_keys") on 1.7.0rc3 and is not served until its keys are persisted again: the index stage's
  full-build path (`read_key_rows`, `keys_from_graph`, `persist_keys`, which embeds only the changed keys), then a
  serve restart. A bonfire with no such message loads unchanged.

### 1.7.0rc2

- **`mk build --genesis`: a bonfire bootstraps from its Purpose alone.** A seed of a few Setup sentences gives zeros
  that five checks of a full build refuse. Each check accepts its zero only in a genesis build (`BuildConfig.genesis`,
  through one predicate, `config.genesis_zero_ok`), and only that zero: graph-A takes zero picks when the recorded
  picks plan counted no message (the forward append's branch, `_no_picks_needed`); the taxonomy census passes with 0
  events; taxq passes with 0 valid and 0 rejected questions; the C2 rows may hold the Purpose node and no Question
  node only when labelq wrote no question (`write.genesis_without_questions`, the forward append's branch of
  `check_c2_rows`; with labelq questions in the C2 keywords, zero Question nodes is still refused, at `build_dry`,
  the write and the rewrite); sanity searches with the Purpose when there is no question. A genesis census with
  events keeps its band, a genesis taxq with questions keeps its minimum. Without the flag every check is as before,
  and a sanity with no question now stops with "sanity: no questions to search with" (it was an IndexError).
- The flag is recorded as `genesis: true` in `build-config.json`, the `driver-start` event's `recipe` and
  `sealed.json`; a build without it writes none of these keys (its files are byte-identical to before). A
  conversation root resumes only with the flag it was started with: the preflight refuses the other one before any
  stage. `BonfireDir.run_config` resets it (`bonfire.NOT_INHERITED`), so an append of a genesis bonfire is not a
  genesis build, and a config that sets both `genesis` and an append is refused. Version `1.7.0rc2`.

##### Upgrade notes

- **A kernel before 1.7.0rc2 cannot read a genesis bootstrap's `build-config.json`** (`unknown build-config keys:
  ['genesis']`): there, `mk bonfire init`, `mk append` and `mk inspect` refuse a genesis bootstrap and its bonfire.
  Roll back only before the first genesis build; a bonfire built without the flag is not affected.

### 1.7.0rc1

- **`POST /v1/bonfires/{id}/entities/sheet`: who is X, with each summary line's source messages.** Body `{"name": str}`
  or `{"uuid": str}`; answers `{"entities": [...]}` (at most 3), each `{uuid, name, kinds, summary, lines: [{text,
  message_ids, episode_ids}], aliases, identities, timeline}`; a timeline row's `happened_at` is null unless its
  `time_source` resolved it, and it carries `asserted` and `modality`; the timeline is read per entity. Names match `n.name` and `identity_aliases` (the predicate of
  `aliased_uuids`), then a `SAME_AS` component reads as one entity with every member's lines. 422 on a bad body, 404
  with an empty list when nothing matches. Reads only; no model call. Version `1.7.0rc1`.

### 1.6.0

A round shard no longer pins the bytes of `ledger.py` and two construction modules. New shards record
`ledger_schema: 1` and open whatever those files hold; a shard made before opens when its pinned digests are in
`ledger.ACCEPTED_CODE_PINS` (the 1.5.x digests), and is refused with "round implementation changed" otherwise.

#### Model snapshots

- **A provider's dated snapshot of a pinned model alias no longer breaks a send.** The gap sender, the Jev archive
  reader and the Jev window validator accept `<alias>` or `<alias>-<digits>` (`model_family_ok`). A paid response from
  another family is refused after its real cost is recorded as spent, on both the gap and the Jev path; the attempt is
  kept as `<name>-refused-<n>` and the wire can be asked again.

#### Per-append budgets and a spend ceiling

- **An append spends against its own budget, not the bonfire's lifetime caps.** The round cap and the integrated cap
  ($25) sealed at the bootstrap bound the bootstrap only. An append's budget is `bonfire.append_budget`: 1.5 x $0.0025
  per new message, or 1.5 x the Jev spend its saved plans project, whichever is larger, floored at $0.10 per session
  (`APPEND_FLOOR_USD`). Its stages open the shard with the limit "the shard's held-or-spent total when the run was
  planned + the budget" (`BuildConfig.append_base_usd`, `ledger.Shard.limit_usd`); a hold past it is refused with
  "append budget exhausted before send". The refusal "bonfire round budget exhausted: refresh the bonfire" is gone.
- **`mk bonfire init --spend-ceiling-usd X`** records `spend_ceiling_usd` in `bonfire.json`
  (`BonfireDir.spend_ceiling_usd`), and no stage takes the shard past it. Before any stage, both preflights
  (`mk append`'s and the driver's) hold the shard's spend plus the run's estimate (`bonfire.append_estimate`: the
  budget with no plans, or the last recorded run's `budget_usd` scaled to this run's new messages, whichever is
  larger) against it, and refuse with "bonfire spend ceiling reached". The estimate is not a bound: a run whose plans
  project more can still stop at the ceiling after earlier stages paid. A `bonfire.json` without the field has no
  ceiling.
- **A shard's sealed caps bind at `reserve`, not at the open.** A plain open (no append limit) of a shard whose spend
  is already past its round cap or its integrated cap opens and replays its archive; a new hold past either cap is
  refused ("shared round budget exhausted before send", "shared integrated budget exceeded: $X exhausted before
  send"). `llm.LedgerView.summary` no longer raises past a cap either.
- **`mk serve`'s `/healthz`** adds `version` (the installed package's) and `spend: {bonfire_id: {charged_usd,
  ceiling_usd}}` as last read. A long-lived shard takes seconds to read, so the probe never waits: a spend whose
  shard's budget dir or `bonfire.json` changed (or older than five minutes) is read again in the background, and a
  bonfire not read yet is `null`.
- A forward append's recorded per-conversation limit (`conv_limit`) is its budget.
- A run's entry in `bonfire.json` records `budget_usd` in place of `worst_usd`.

##### Upgrade notes

- **A 1.5.x kernel cannot open a shard that a 1.6 append took past its sealed round cap** ("shared round budget
  exceeded"). Nothing in the shard's manifest changed (`ledger_schema` stays 1); roll back only before that point.
- **Opening a bonfire's shard outside an append** (`mk ask` on a run root or the bootstrap root, a resumed bootstrap
  build) still holds new sends to its sealed caps. Once appends take the shard past its round cap, such an open replays
  what is archived, and a frame that needs a new send is refused; `mk ask` has no frame fallback, so that question
  errors. `mk serve` and the library API ask through the query shard and are not affected.
- **A bonfire recorded only by 1.5.x** has no `budget_usd` in its runs, so its ceiling estimate stays at the budget
  with no plans until a 1.6 run is recorded.
- `bonfire.budget_problems` takes `ceiling=` and `runs=` (default: the `bonfire.json` beside the shard), and
  `BonfireDir.run_config` takes `base_usd=` (default: the shard's spend now).

#### Appends on a shared Neo4j

- **Other groups' writes no longer fail a write.** A write checks only its own group: the writer's foreign-uuid
  refusal and, in an append, the per-uuid readback of the rows it wrote (a node uuid or `(type, uuid)` edge the write
  sent and the group does not hold afterwards fails the append gate: `append.readback`). The snapshot of every group's node and edge
  counts (before, during and after the write; "other groups changed") is taken only under
  `BuildConfig.strict_db_snapshot` (default off: the bench's strict mode), and `--concurrent-group-prefix` applies only
  there, so a tenant writing the same Neo4j at the same time no longer stops an append. A write result records
  `strict_db_snapshot` and carries `snapshot_*` / `other_groups` only when it is on.
- **A clone is refused up front.** A uuid is a global identity. `mk append` and `bonfire.relocate` (so every store
  hydrate) refuse a bonfire whose chain entities do not carry the uuid its group gives their key: "chain minted under
  another group (a clone)" (`forward.check_chain_group`; `forward.Refused` is a `ValueError`, so `mk append` exits 2).
  Such a copy kept its source's uuids under a new group, and its first append that reuses an entity blocks both.
- **The foreign-uuid refusal names the group.** `kg_writer.write_rows` raises `kg_writer.ForeignUuids` (a
  `ValueError`): `.rows` holds `(uuid, owning group)`, the group None for an edge endpoint in neither the batch nor the
  group, and the message names the owning groups.
- **Typed lookups only.** The writer's endpoint check and `read_group` read per node label instead of scanning every
  node. The kernel's own edge types (`NEXT_SAGA`, `ANSWERS`, `ANSWERS_VIA`, `DERIVED_FROM`, `INVALIDATED_BY`,
  `SAME_AS`) get uuid range indexes (`next_saga_uuid`, `answers_uuid`, `answers_via_uuid`, `derived_from_uuid`,
  `invalidated_by_uuid`, `same_as_uuid`), issued `IF NOT EXISTS` by the next write; no index is dropped or rebuilt.

##### Upgrade notes

- `stages.write.write_problems` takes a required `strict=`.
- `strict_db_snapshot` has no CLI flag: set it on the `BuildConfig`.
- **Rollback to 1.5.x.** A `build-config.json` carries `strict_db_snapshot`, `append_base_usd` and
  `spend_ceiling_usd` only when they are set (on, or not None), and a 1.5.x kernel refuses a config that carries
  one; every other 1.6 config is readable by 1.5.x. A shard first made under 1.6 (a query shard created after the
  upgrade, say) is refused by 1.5.x with "round implementation changed": it pins the 1.6 bytes of `ledger.py` and the
  two construction modules, and 1.5.x re-hashes every pin against its own files.
- The first 1.6 write on a database creates the six indexes; Neo4j populates them in the background.

#### A partly built session

- **`mk append --on-built skip` (`append_session(..., on_built="skip")`) appends a partly built session's new
  messages.** A session holding a message id an earlier session built (an edited message that kept its id) is refused
  by default ("session overlaps an already-built session"), and the sessions after it wait. With `skip` its built ids
  are dropped before anything is built or spent: the rest keep their order and timestamps, the session is dated by the
  first of them and the budget counts them alone. The run's entry in `bonfire.json` and the results of `bonfire.append`
  and `append_session` carry `skipped_ids`; the run pins them in its forward input (`forward.json`), so an entry
  recorded by the crash repair (`repair_unrecorded`) carries them too. The inputs keep the session as given. A wholly
  built session is still a `duplicate`, and the default (`refuse`) is unchanged. A stopped run is sent again with the
  same `on_built`.

##### Upgrade notes

- `append_session`'s result always carries `skipped_ids` (empty unless the run skipped ids), `duplicate` and `empty`
  included; for a run, its `sessions`, `new_messages` and `skipped_ids` are those of `bonfire.append`'s result, which
  adds `session_number`. `BonfireDir.plan_session` and `bonfire.append` take `on_built=`.

#### Runtime

- **A frame part anchors on its form's verb.** `frames.frame_parts` takes the first verb of a card's or taught
  phrase's form, as the turn parser tags it, as its lemma ("please send {X} to {Y}" is `send`); a form with no verb
  takes its first noun or proper noun ("the {X} report" is `report`, "{X} for review" is `review`). A form with
  neither ("{X} please", "who {X}?") gets no part and is read by its whole form only. The first word of a form
  ("the", "please") no longer affords a phrase on nearly every turn. `constructions.literal_tokens` gives the
  tokens of a form's literals with their POS, the slots' stand-ins left out.
- **The text grammar boots on any pinned FCG Go.** `kernel-ext.lisp` serves the unit grammar only.
  `RuntimeEngine.load(tiers=)` checks the fcg-go pin against `KERNEL_EXT_IMAGE` before it touches the engine, then
  sends the extensions with the unit grammar (again after a restart). An engine start no longer installs them or
  refuses a pin they were not tested against, and a refused unit grammar leaves the text grammar running.
- **A plan the runtime cannot read is logged.** `mk serve` logs a runtime that does not start or whose store does not
  load (`runtime.plan_unavailable`) and one that raises (`runtime.plan_failed`) before it answers the unread plan
  ("runtime unavailable: ...", "runtime error: ..."). The log carries the plan's reason, secrets redacted, never a
  traceback. All engines in use is not logged.
  The warnings for an engine that would not stop and a graph reader that would not close carry the same redacted
  reason, and the question-parse warning carries the exception type only.
- **A runtime forgets the chats it has no plans for.** Once none of a chat's plans remain, `Runtime` drops its
  latest-plan mark and its turn lock (the lock only while no turn of the chat holds or awaits it).
- **The spaCy parse caches are bounded.** `g1_projection._parse`, `possessors`, `analyse` and `_short_name`,
  `g1_mentions._own_doc` and `g1_nlp.lemma_key` keep the 4096 latest entries (`lru_cache`). The worker never
  restarts, and each kept every text it ever parsed.

##### Upgrade notes

- A phrase whose form holds no verb and no noun ("who {X}?", "{X} please") is no longer afforded by a frame part.
  A complete reading through its whole form is unchanged.
- A part's id follows its anchor (`frame.<name>.<word>`): a phrase that opens on a determiner, a preposition or a
  slot gets a new part id ("the {X} report": `frame.<id>.report`, not `frame.<id>.the`). Parts are never stored, so
  nothing migrates.

### 1.5.2

Compose any tool: a taught phrase's steps may be calls to any catalogue tool (an agent's tool, `think`), a tool says
where its step runs, and an untyped result fills any string parameter.

#### Upgrade notes

- **A catalogue row with an `x-runner` other than `office` or `agent` is refused** (it was ignored). Rows without it
  run in the office, as before.
- **A construction store holding a row with id `frame.editor.phrase` no longer opens:** it is a base id now. Rename
  such a row before upgrading.
- **"phrase" anywhere in a turn affords the editor** (`teach_phrase`), so a turn mentioning a phrase now carries an
  editor frame beside its others.
- **Bodies 1.5.1 refused on kind now stage.** "it" or "them" into a kinded parameter with only untyped results before
  it (`look at Doing`, `take it`) binds the untyped result; the tool checks the value when it runs. Where a producer of
  the parameter's kind comes before, it is still the one bound.
- **"it" after several untyped results now stages.** Where every fitting producer's tool has no `x-returns`, a
  sentence's "it" is the nearest of them: in a catalogue without `x-returns`, "take water, take herb, give it to {R}"
  gives the herb's take (1.5.1 refused it: "say which"). A body 1.5.1 accepted binds as it did.
- **A teach body entry may be an object.** `TeachRequest.body` is `list[str | dict]`; a caller reading
  `provenance["request"]["body"]` back may find structured steps in it.

#### Structured steps

- **`{"call": "<tool>", "args": {"<param>": "<value>"}}` beside clauses** (`teach._call_step`). A value is a slot
  (`"{X}"`), `"it"` (the result of the step just before), `"them"` (as in a clause) or a literal. The step compiles to
  the ops a read clause of the tool gives (`procedure` + `seq` + the tool op, literals as `const` ops), so the self-read,
  the duplicate check, slot rebinding and plan-time expansion are unchanged. A tool with no `x-forms` composes.
- **Checked against the catalogue:** an unknown tool or a keyword, a parameter the tool does not have, one left out
  (a step gives every parameter the tool declares), a slot no form has, "it" with no step before or one that produces
  nothing, "them" into a parameter that takes one value, a value mixing a slot and text, a malformed call: each a
  problem naming it. `POST .../teach` takes object body entries; one of another shape, or past the caps, is a 422.
- **An agent step after the board is refused (C12):** a step whose tool runs in the agent after an office step that
  waits for the board's approval, in the expanded program, could never run once the board approves.

#### Runners

- **`x-runner: office | agent`** on a catalogue row (default `office`), kept as `ToolSpec.runner` and
  `OpSpec.runner`. The kernel uses it only for the C12 rule; the caller routes pending steps by it.

#### Untyped results fill string parameters (C13)

- In teaching, where no earlier step has a kinded parameter's kind, an untyped result fits it (typed producers first).
- **A sentence's "it" after several untyped results is the nearest of them**, as a structured step's "it" is the step
  just before: "web search for X, think: pick the most urgent, write it" (or "make a ticket from it") stages and takes
  what think gave (it was refused: "it could be the result of ... or ..."). Where a fitting producer's tool has
  an `x-returns` (typed, or kind-less as `take` in 1.5.1 bodies), several fitting producers are still refused ("say
  which"), and the refusal now also says a structured step's "it" is the step before.
  In live planning nothing changed: a text value in a kinded parameter was never refused (only a resolved entity of
  another store kind blocks), now pinned by a test.

#### The editor

- **`frame.editor.phrase`**, a base part of "phrase" (read as the plain token and as the marked verb), affords
  `teach_phrase`: "make a phrase for that" lets the agent draft a phrase from its own context.

#### Tests

- A phrase whose tool a later catalogue no longer offers (the agent's source gone) plans as a refusal naming the tool,
  never a traceback (Review Focus 5).

### 1.5.1

Affordances: every plan says what read which words of the turn, so the agent can act on partial readings; a
follow-up composes with the open frame it answers; teaching is admission, not a sweep.

#### Upgrade notes

- A verb inside a bound role affords its frame too (live 10-23: "search about the monkey situation and write note
  about it" read as search alone): a turn with two marked verbs or more reads each bound role of its complete frames
  on its own, one level deep (`Runtime._slot_frames`).
- **A construction store holding a row with id `reply.yes`, `reply.no`, `reply.yes.period`, `reply.yes.exclaim`,
  `reply.no.period`, `reply.no.exclaim`, `frame.editor.teach` or `frame.editor.mean` no longer opens:** those are
  base ids now. Rename such rows before upgrading.
- **A bare "yes" or "no", alone or with a final "." or "!", reads as a reply root.** The plan stays `unread` (nothing
  runs), now with a `reply` frame.
- **Teaching no longer refuses a generalisation.** A form broader than a live construction ("who {X}?" next to the base
  `who-be-x`, "give {X}" next to a "give {X} to {Y}" card) is staged, and once confirmed it is the primary reading of
  the turns it reads, the other construction's reading coming as a competing analysis (`frames`). A caller that
  relied on the refusal to protect a construction now gets both readings and chooses.
- **`replace` removes same-skeleton rows only.** A 1.5.0 replace also removed the taught or mined rows a form
  generalised; a 1.5.1 replace removes only the live rows with its skeleton. A draft staged by 1.5.0 keeps the
  `replaces` it recorded and is confirmed with them.
- **`provenance["shadows"]` is no longer written,** and **a confirm no longer re-reads anything**: it checks the
  teacher, the drafts it waits for and the rivals gone live since, then goes live. Rows that carry `shadows` keep it.
- **A teach that duplicates live constructions is refused** with `form '<form>' is already the live construction
  <id>` per form (and nothing else), where 1.5.0 said it competed and asked for `replace`. A draft only some of whose
  forms are live as they are is taught as before.
- **A teach whose form competes now also reads its body**, so its `problems` hold the competition and the body's
  problems together.

#### Frames

- **Every plan carries `frames` and `left`.** A frame is `{name, kind, construction, roles, complete, analysis,
  composed_with}`; `left` is the text the primary analysis did not consume, as written. Unread plans carry them too.
- **A frame part per leading word of every live card and taught phrase** (`runtime.frames.frame_parts`; a form that
  opens on a slot gets its first literal, whole: "{X} for review" -> "for review"), generated when the grammar loads
  and loaded below everything live: "make onboard josh" reads a partial `make` frame, and each card still reads its
  own sentences completely. A part is never stored and records no standing. A word that leads a card and a taught
  phrase affords both: only one part can consume the word, so every part sharing its trigger reports its frame
  ("please take the new one" with a taught "take {X} for {Y}" gives `take` and the phrase).
- **A part reads whole tokens only.** The compiler anchors a part's outer edges on token edges (a space, or a verb
  mark), so "archive" never reads inside "archives" or "⟪archive⟫" and a taught "who {X}?" never reads inside
  "whose" or "whole"; `left` names each unread word once ("please archive it" leaves `please` and `it`). A one-word
  part also reads its word as written as a marked verb: "archive" alone is a noun, but "please archive it" and "it
  was archived" hold the verb, and afford `archive`. Two parts on adjacent plain words share the space between them,
  so only one of them reads.
- **Frame parts are searched one after another.** A search node gets at most one child made by a frame part, so a
  message's parts apply in one order, not every order and subset: ten clauses with a verb each read in about 0.1 s
  where they took 1.1 s (530 nodes), and the plan in under 0.5 s where it took 2.5 s; every frame is still read.
- **Up to three competing analyses.** A complete reading adds the other analyses the engine finds
  (`RuntimeEngine.comprehend_all`, one per root construction): "who is Zed?" with a taught "who {X}?" gives the phrase
  and the base memory question as two frames. The search goes on past one root's filler paths ("the report" through
  `np-the` and `np-bare`) until three roots, the end of the search, 32 more solutions or the engine's deadline, so a
  determiner never hides a competing card. The continued search leaves frame parts out (no complete analysis holds
  one) and checks the deadline at every node, so it ends inside the engine: "make a ticket from" a title holding
  eight verbs costs the n-best about as much as one read (it cost a second).
- **The editor and the replies are base rows.** `frame.editor.teach` (form "teach") and `frame.editor.mean` (form
  "means") afford `teach_phrase`. "means" alone normalises to the marked "mean" ("X means Y", "when I say X I mean
  Y"), and as a one-word part it also reads the marked "means" the turn parser keeps in some sentences ("file X means
  make a ticket from X": tagged VBP, the lemma stays "means"); a plain "mean" part missed that one. `reply.yes` and
  `reply.no` are narrow roots, with one row per final mark ("yes.", "Yes!"). There is no catch-all root.
- **The competing analyses never cost the primary its plan.** When the n-best read fails, the primary reading stands
  alone and `reading["alternatives"]` says why.
- **An unread plan always says what was left.** When the engine fails a turn, its plan's `left` is the whole turn, as
  on every other path where nothing read it (it was empty).

#### Context

- **The plan takes the open frames of the turn before** (`Runtime.plan(..., context=[Frame])`, HTTP: `context` in the
  plan body, at most 8 Frame objects; a ninth, a frame with no name, an unknown key or a wrong type is a 422).
- **A bare fragment composes with the open frame it answers.** A turn that reads nothing complete and has no frame of
  its own is read again inside each open frame's own forms, the fragment in the open slot ("JUJ-3" against an open
  `take` is read as "take JUJ-3"); the first rendering that reads completely through a root of that frame is the
  plan's reading, its frames carrying `composed_with` and `reading["composed"]` naming the rendering. A root with
  another slot still open is never rendered, a complete turn and a turn with a frame of its own are never composed,
  and the caller sends only frames whose effect has not run. The fragment's final marks are dropped first, so
  "JUJ-3." is "take JUJ-3" too; a closing quote or bracket (spaCy morph `PunctType` Quot or Brck) is no final mark
  and stays with its opener ('"JUJ-3"' binds '"JUJ-3"'); a turn of nothing but marks ("?" in answer to "which
  ticket?") composes with nothing.
- **A frame the agent held back is re-filled by the answer.** After every open role, a frame whose one bound role is
  a one-slot root's is read again with the fragment in place of that value: "take the new one" reads completely
  (take(ticket="new one")), the agent asks "which ticket?" and "JUJ-3" composes as take(ticket=JUJ-3). A frame with
  two or more roles bound is never re-filled.

#### Teaching

- **Teaching is admission.** A teach reads its body clauses and its forms' own renderings only: once before the draft
  loads and once with it on top (the self-read). The sweep goes: no live root is rendered and read, in `teach` or in
  `confirm` (it read 3,902 texts for "look at the office", about 2 minutes, holding the engine). Measured on the
  office catalogue (trf parser), each the first teach of its shape in a process that has just loaded its grammar:
  "look at the office" (five body clauses) 0.5 s, "file {X}" 0.5 s, "triage {X} for {R}" 0.55 s, "standup" 0.4 s; a
  confirm 0.1-0.25 s. A teach never parses a live form again: compiling the grammar (`load_store`, at start and on
  every reload) parses each form once, and the competition and duplicate checks cut the live roots' skeletons from
  that parse (only roots whose slots sit where the draft's do are compared).
- **The draft says what it competes with.** `Draft.competes` (HTTP: `competes` in the teach response) maps each form's
  text to the frames the live grammar reads that form's own renderings as before the draft: "look at the office" gives
  the `look` card with `place` "office"; a form nothing reads gives `[]`.
- **A duplicate is refused naming it:** a draft every form of which is already live as a taught phrase or a catalogue
  card with the same normalised form and the same meaning up to slot and variable names.
- **Competition is settled at read time.** Broad phrases are admitted; where two constructions read a turn, both reach
  the agent as frames and the call's outcome moves their standing (`store.record_uses`).

### 1.5.0

Composition: a taught body can use an earlier step's result ("it", "them"), and "them" and "everyone" are in the base set.

#### Upgrade notes

- **`create_app(warm=False)` answers `/healthz` 200 again.** Library and test callers that build the app without the
  warm-up get `200 {"status": "ok", ...}` whatever is open, and a bonfire not open yet is listed as `null`, as before
  1.4.1 (1.4.1 listed it `warming` and answered 503 until a question opened it). `mk serve` (warm) is unchanged.
- **`gather` and `plural` are reserved op names.** A catalogue tool named either fails at registry build; rename it
  before upgrading.
- **A live "everyone" distributes the turn's procedure once per member,** like a taught one, unless it fills a `many`
  parameter. A live turn that said "everyone" before ran the procedure once; it now runs once per member.
- **A construction store holding a row with id `np-them` or `np-everyone` no longer opens:** those are base ids now,
  and a base id is refused when a stored row is admitted. Rename such rows before upgrading.
- **"them" and "everyone" are no longer name text.** In 1.4 they read as names (the bare filler); now "them" is a
  reference to earlier steps (a repair in a live turn) and "everyone" is the turn's participants.
- **A turn saying "everyone" with no participants now repairs.**
- **A memory question with "everyone"** passes the participants to `find-entity`, which finds nothing: the result is
  empty, with no error.

#### Composition

- **A tool may declare which parameter its result stands for.** The catalogue key `x-returns: {"of": <param>}` reads
  into `ToolSpec.returns` and `OpSpec.returns`.
- **"them" and "everyone" are in the base set.** Base rows `np-them` and `np-everyone`, a `plural` marker and a
  variadic `gather` op. "them" in a live turn (no earlier step to point at) is a repair.
- **Teaching binds "it" and "them" to earlier steps' results.** A slot an annotated tool produced means its result
  afterwards. Inside a phrase that `everyone` runs once per member (a taught "brew tea for everyone"), a reference
  gathers within each member's turn.
- **A reference to a step that runs once per member is refused,** and its slot is not rebound.
- **A reference ("it"/"them") into a `many` parameter is always a list:** "it" is a list of one, as "them" with one
  result is. Plain slot or constant text into a `many` parameter stays text, as in 1.4.
- **An "everyone" that fills an invoked phrase's slot is decided by what that slot feeds,** in a taught body as in a
  live turn: a taught "stock everyone", where "stock {R}" makes tea from R, makes once from all the participants.
- **One "everyone" read into both a `many` parameter and a one-value one is refused:** at teach time a problem, in a
  live turn a repair ("say it in two clauses").
- **Replay needs `committed` in issue order.** Entries with the same tool and args resolve in the order given, so the
  game passes its ledger rows in the order the steps were issued; a phrase run once per member then replays each
  member's own results.
- **Replay holds under composition:** a composed phrase replays with identical args. The contract docs
  (`docs/runtime-contract.md`, "References in a taught body") and the README describe references and `x-returns`.

#### Serving

- **A bonfire sealed after its warm-up looked is warmed.** One listed `failed: no sealed run yet` is checked again
  each time `/healthz` lists it, and opened once it has a sealed run.
- **A Neo4j host name that does not resolve yet is retried.** The warm-up treats any name-resolution failure as
  transient, including the ones the driver raises as a `ValueError` (for example `EAI_FAIL`), not only the ones it maps
  to `ServiceUnavailable`.
- **A cache-only call waiting on a paid send that fails gets a cache miss** (recorded), not the provider's error.
- **A paid call never sends through a shard ledger opened by a cache-only call or with another key.** It is refused by
  name at the send, before anything is reserved; its replays still answer.
- **Correction to 1.4.1:** the one-time WordNet readiness probe (`wordnet_ready`) runs under the shared lookup lock,
  not a probe lock of its own (the 1.4.1 entry's "own probe lock" line is superseded).

#### CLI

- **`mk bonfire backfill-lines`** counts a node with summary text and no line ids as `skipped` and names why in
  `skipped_reasons` (the readers keep it legacy, so the group's backfill note stays); a node with no line ids and a
  blank summary holds no line and is not counted.

## 1.4.1 (2026-10-21)

Serving parses only the question. Nothing stored is parsed at question time, and `mk serve` answers `/healthz` 200 only
when it is warm.

### Upgrade notes

- **Graphs built before 1.4.1 need `mk bonfire backfill-lines <bonfire>` once.** Until then their sheets are
  recency-only, and the kernel logs one warning per group per process that names the backfill (the sheet text carries
  no note).
- **Deploy order: stop every 1.4.0 appender (the kernel worker) before running the backfill, and deploy 1.4.1 before
  restarting it.** A 1.4.0 append (or a rollback to 1.4.0) after the backfill grows a node's line ids but not its line
  texts. 1.4.1 reads such a node as not backfilled (recency only) and appends no line data to it; running the backfill
  again repairs it.
- **`/healthz` answers 503 while `mk serve` warms.** A healthcheck or a client that reads `/healthz` should treat 503 as
  "starting", not as down.
- **Appends from 1.4.1 write the new per-line properties** (`summary_line_texts`, `summary_identity_line_ids`). A stored
  summary without whole line data stays that way on append; it is never made partial. That covers a node built before
  1.4.1 and one desynced since its backfill (texts shorter than the line ids, or one list without the other).
- **spaCy parses in one process no longer overlap.** Every pipeline the kernel loads takes one process-wide lock per
  `nlp(text)` and per `nlp.pipe(...)`, because a transformer pipeline run from two threads at once can parse
  differently from a serial run. A key rebuilt at open must equal the stored one. Separate processes (capture shards)
  still parse in parallel. Parallel questions in one process (any bonfire) queue on the lock for their question parse
  (about 35 ms each on trf).

### Summary lines

- **A summary line carries its text and its identity at write.** A node holds `summary_line_texts` (parallel to
  `summary_line_ids`; joined, they are `summary`) and `summary_identity_line_ids` (the identity lines, in line order). A
  forward append carries `summary_append_identity_line_ids` beside `summary_append`.
- **Identity comes from structure, by referent.** A line is an identity line when its clause is asserted and actual and
  the subject of a copula with a noun complement ("Kira Dace is a pilot") lies in the bearer's own words on the
  message's parse. Under `structure` these are the bearer's mention tokens; under `phrase`, or for a referent with no
  mention on that parse, they are the role's capture span, realigned when the capture indices are shifted. A
  coordinated subject ("Kira Dace and Milo are pilots") makes the line an identity line of each conjunct. Names play
  no part. A copula subject outside every bearer's words is counted (`identity_subject_outside_bearers` in the
  projection's routing).
- **`codings_anchored` checks the data:** the lists are parallel to the line ids, the texts re-join to `summary`, and the
  identity ids are a subset in line order, for base and appended rows. A node's texts and identity ids are present or
  absent together.

### Serving

- **Sheets and the runtime identity ops read the line records.** No stored text is parsed at question time. A graph
  without the data gets recency-only sheets and one logged warning per group per process naming the backfill; a
  `SAME_AS` sheet with a member that lacks the data logs it too. An entity with no lines (a blank summary, no line ids)
  has data: it never triggers the warning.
- **`/context` parses the question once** (pinned by a test, for a question with a subject). The search over the
  preloaded index runs once, in a worker thread, and serves both the context and `retrieved`; Neo4j reads await on the
  loop and assembly runs off it. A question with a newline or a double space is no longer parsed twice.
- **Message weekday notes are computed once,** when a run opens, not per question. A message with a bare weekday is
  parsed twice at open (once for its key, once for its note). One-shot callers open a server per call (`mk ask`,
  `answer.answer`, `api.acontext`), so they now compute every message's note at each open. Build and the harness
  compute them as before. Output is unchanged.
- **One lock for every WordNet synset reader.** NLTK reads an uncached synset by seek and readline on one shared
  handle, so two parallel questions could read each other's line. The synset lookups (`grammar.wordnet_semantics`,
  `graph.identity.is_common_noun`) now share one lock, and alias matching runs in a single worker-thread hop. The
  one-time readiness probe (`wordnet_ready`) runs under its own probe lock. The
  `morphy` readers (`runtime/derender.py`, `envelope/render.py`) do not take it. They read NLTK's in-memory exception
  and index maps, not the shared file handle.
- **A shard's lock guards its ledger, not the network.** Parallel new questions send their frame calls in parallel,
  with one send per wire.
- **`mk serve` is ready when warm.** The models (parsers, WordNet, the token encoder) and every bonfire open at
  startup. `/healthz` answers `503 {"status": "warming", ...}` until every bonfire is warm or failed and at least one
  is warm (a transient Neo4j failure retries), then 200. A bonfire discovered after that is listed `warming` while it
  opens, and `/healthz` stays 200.

### CLI

- **`mk bonfire backfill-lines BONFIRE_DIR [--dry-run] [--neo4j-uri URI] [--neo4j-user USER]`** writes the per-line
  properties to a graph built before 1.4.1. Line texts are recovered by id hash, the identity flags are parsed once
  offline, and only the two new properties are written. It also repairs a node desynced since (see Upgrade notes). Such
  a node keeps the flags of the lines it already held, and only its other lines are parsed. The flags depend on the
  host's query parser, so the JSON output names it (`parser`: model, version, fallback note). `--dry-run` counts and
  writes nothing; the Neo4j target defaults to the bootstrap's.

### Fixes

- **A reused stored instance is not a duplicate name** in `build_dry`. The check read the chain's minted key, and reused
  rows carry no `name_key`.
- **Under the warm capture slot the preflight does not check the managed parser's port,** because the slot's parser
  holds it.

## 1.4.0 (2026-10-21)

### Runtime grammar

- **A second grammar reads every incoming turn.**
  - It is a small FCG grammar of effectful constructions, on its own deterministic engine (the pinned FCG Go on
    `MK_RUNTIME_FCG_PORT`, default 9711).
  - It turns a question or a request into a program of memory operations and tool steps, in milliseconds and with no
    language model.
  - The extraction parse of build and append is unchanged.
- **The shipped base set.** `data/runtime-grammar/en/base.json` holds the closed-class grammar as constructions: wh and
  polar roots, noun phrases, deixis (I, me, you, it, that one) and time fillers.
- **The shipped question map.** `data/runtime-grammar/question-map.json` says which role of a frame answers who, what,
  where or when. It is generated from the PropBank frame files' function tags. A role whose VerbNet theme names a
  store kind (a location is a Place, an employer an Organisation) carries that kind.
- **State answers for "where does X V?".** When X's events give no answer, the answer is the entities that X's state
  lines name, of the kinds the question's roles expect only: "where does X work?" reads the organisation, "where does
  X live?" a place, and a line with no entity of the expected kind stays a repair.
- **Keywords.** The operations a program may use are a closed set of keywords with rules text: value keywords, read-only
  memory keywords over the bonfire's group, and lexicon keywords. [docs/keywords.md](docs/keywords.md) is generated
  from the code (`mk keywords`), and a test fails when it is stale. Teaching and remining compose keywords and never
  add one.
- **`Runtime.plan(turn) -> TurnPlan` and `Runtime.resume(plan_id, results)`.**
  - A turn becomes `answered`, `needs_tools`, `repair` or `unread`.
  - A program runs only when the reading is complete, validates and leaves no constraint unapplied. A program that
    finds nothing (a name that names nobody, a verb that names no frame, an empty answer) is a repair with
    diagnostics, never a guessed answer. An empty-result repair also carries the fallback `recall`.
  - The plan's `hand` lists the constructions the turn drew, with their standing.
  - "it" and "that one" read the chat's turn window.
  - The full wire contract is [docs/runtime-contract.md](docs/runtime-contract.md). The shapes below are new in 1.4.0.
- **TurnPlan shape.**
  - `tool_steps[]`: `step_id`, `tool`, `args`, `side_effect`, `needs_confirmation`, `actor` (the turn's speaker) and
    `checks` (`[{param, constraint, result}]`).
  - `failure` (`{step_id, tool, kind, detail}`) is set on a repair that a verdict caused, and `licence` rows carry
    `needs_confirmation`.
- **Result rows.** A resume row is `{step_id, tool, args, verdict, value, detail}`, with `verdict` one of `committed`,
  `precondition_failed`, `tool_error` and `declined`. A row whose tool or canonical-JSON args differ from the pending
  step's is refused (`ResumeError`) before any state changes. A pending step's `step_id` is
  `<plan_id>:<i>:<occurrence>`, so no id is valid in two plans.
- **Commissive reading.** The envelope's commissive reading takes its future auxiliaries from `tense.json`, so "shall"
  counts like "will" ("I shall go", "Shall we repaint the shed this weekend?" read commissive).
- **`/healthz` carries `parser_notes`** (`{"query": "sm-fallback"}`) while the query parser has fallen back to the small
  model.
- **The licence.** `licence` is `null` only on an `unread` turn and on a repair of a program that names no tool (a memory
  question, which falls back to today's path). A program that names any tool, or contains an `invoke`, carries its own
  licence on every outcome: a plan-time repair, a verdict repair, a repair after a committed effect. A dangling
  `invoke` repairs with at least `[]`. A tool a taught row copied inline and the catalogue has since dropped is still
  listed, with `side_effect` unknown and `needs_confirmation` true.
- **Slot constraints.** A catalogue parameter's `constraint` (a WordNet concept) or `kind` (a store kind) is checked when
  a step is issued. A lexical check never blocks: its `pass`, `fail` or `unknown` result rides on the step as `checks`
  for the game's Resolve. Only a store-kind mismatch on a resolved memory entity blocks (a repair, licence kept). This corrects amendment G3.
  Generated catalogue cards bind slots as text and never resolve entities, so a catalogue `kind` does not block today;
  the block applies to hand-written cards whose meanings resolve entities. The workshop plan decides whether kind-typed
  parameters resolve.
- **Standing separates the world from the construction.** A row's `standing` gains `uses_world`: a repair the room
  caused (a verdict `precondition_failed`, `declined` or `tool_error`) counts there and never in `uses_ok` /
  `uses_failed`, so precedence is unchanged by it. A misreading, an empty result or an unresolved name still counts as
  a failed use. `ConstructionStore.record_uses` takes the outcomes `ok`, `failed` or `world` (bools are refused).
  **Downgrade:** a `constructions.jsonl` written by 1.4.0 holds `uses_world`, which earlier kernels reject; rows written
  before 1.4.0 read as `uses_world` 0.
- **Retry and committed effects.**
  - `Runtime.retry(plan_id)` re-issues the step a verdict refused (status `needs_tools`, the same `step_id`, earlier
    committed results kept) and clears the plan's `failure`. The retry records its own outcome on the plan's
    constructions (a refusal is `world`, a success `ok`). A plan with no `failure` is a `ResumeError`.
  - `Runtime.plan(turn, committed=[{tool, args, value}])` names effects already done. A tool step with the same tool and
    canonical-JSON args (entity dicts included) is resolved with that value (trace `reused`) and never issued; each
    entry resolves one step. Every entry needs a `value` key (`null` when the effect produced none): an entry without
    one is refused (422 over HTTP).
  - A confirm that removes a row by generalisation (no successor) while other rows still `invoke` it carries a warning
    naming those rows (`provenance["warnings"]` on the confirmed rows; `warnings` in the HTTP confirm response).
  - A retry that fails part-way leaves the whole plan as it was, and a problem message names a step by its exposed
    `<plan_id>:<i>:<occurrence>` id.
- **Confirmation policy.** `needs_confirmation` comes from a `ConfirmationPolicy` (`Runtime(policy=...)`, default
  `DEFAULT_POLICY`) for every tool step and licence row: a tool with no side effect is never marked; an unknown side
  effect always is; a tool with `consent: "speaker_turn"` is not marked when the step's `actor` is the turn's speaker;
  every other tool is marked.
- **Composition by reference.** The keyword `invoke` (the only keyword 1.4 adds) lets a taught body clause that reads
  through a taught construction store a reference instead of a copy. The reference expands at plan time into the
  construction's live version (through `provenance["replaced"][].successor`), nesting at most 8 deep. A cycle is
  refused at teach time. Confirming a replacement that removes a row without a successor orphans its invokers, and a
  dangling reference is a repair that records `failed` on the invoker.
- **The tool catalogue.** A bonfire's `runtime/catalogue.json` (`kernel-catalogue/v1`) gives each tool `x-side-effect`,
  `x-args` (per parameter: `role`, `kind`, `constraint`, `many`), `x-consent` (`confirm` or `speaker_turn`) and
  `x-forms`. Each form generates one live hand card, `card.<tool>.<n>`, held in memory beside the stored rows and never
  persisted. A missing annotation is refused.
- **The kernel never runs tools.** Tool steps go to the agent runtime, which resumes the plan with their results.
- **Teaching.** `Runtime.teach` drafts a construction from forms and body clauses, with no language model. The draft is
  usable by its teacher, in its chat, until the teacher confirms it with `Runtime.confirm`; it is then live
  bonfire-wide. A form that would generalise a live construction is refused unless replacing it is asked for, and
  never over the base set. Constructions live in `<bonfire>/runtime/constructions.jsonl`, with each row's standing.

### Parser models

- **One turn parser.** Everything said in a turn (questions, commands, taught phrases) is parsed once, by
  `MK_TURN_SPACY_MODEL` (default `en_core_web_trf`).
- **Serve-time index keys read on the query parser.** `legacy-env` index keys are rendered at serve time with the
  `query` policy (the small model, with a note, when the configured one will not load); the build keeps `extraction`.
  The fallback policies survive any load error, not only the missing-model ones.
- **Extraction on `en_core_web_trf`.** Extraction (build, append and the parser service) defaults to
  `MK_EXTRACTION_SPACY_MODEL=en_core_web_trf`; `MK_EXTRACTION_SPACY_MODEL=en_core_web_sm` opts out. Extraction needs
  `memory-kernel[parser]` and the trf model (`mk setup` installs it) and fails loudly without them, never falling back to
  sm. The closed-class anchors are unchanged: one set serves both parsers. On a 1,778-message chat bootstrap rebuilt on
  trf, entities came out within 2%, kind precision was 0.78 vs 0.75, and a blind per-message comparison of the 335
  messages whose events changed favoured trf 36 to 30 (34 equal). On CPU a build's projection stages take noticeably
  longer (that build spent about 27 minutes more in build_dry and write); `MK_SPACY_GPU=1` with the `gpu` extra moves it
  to a GPU, and a parse-once fix is planned for 1.4.1.
- **The extraction model is recorded.** The capture manifest, the syntax manifest and the build's `driver-start` recipe
  now carry the extraction model's name and installed version. A capture resume and `mk append` (and the library
  `append_session`, so the worker too) refuse when the current model's name differs from the recorded one, naming both and
  the two ways out. **Upgrade:** Bonfires built before 1.4 were parsed with `en_core_web_sm`: set
  `MK_EXTRACTION_SPACY_MODEL=en_core_web_sm` to append to them, or pass `--allow-parser-change` (the keyword
  `allow_parser_change` on the library) to mix parsers knowingly. A version difference alone is only a note. Sealed
  provenance is never rewritten: an older record without a model reads as `en_core_web_sm`.
- **Trf readings.** A restrictive proper-name appositive ("the journalist Milo Dace") is the name, with the noun as its
  kind, a bracketed one ("a mapping tool (Brask Labs)") included; a comma-set one ("my graph, Kira") keeps the
  common noun as the mention and the name as an inner name. The number and person of a pronoun come from
  `data/pronouns.json`, never from the model ("yours" is singular).
- **Subject frame.** A turn with no subject that does not open on a verb is read as addressed ("you ..."), so a bare
  command such as "hum a tune" parses verb-first; the frame word is never the actor and never appears in the text or
  offsets.
- **Fallback.** A turn model that fails to load falls back to `en_core_web_sm` (the frame still applies) with a logged
  warning and `parser: "sm-fallback"` in the TurnPlan trace. `extraction` fails loudly instead.
- **`en_core_web_trf` is in the default setup.** `mk setup` installs the pinned component (436 MB) and `--check` reports
  it not OK when it cannot load. It needs `memory-kernel[parser]` (`spacy-curated-transformers` joins that extra); on a
  base or `[serve]` install, where trf cannot load, `mk setup` skips it with a note naming the extra and the sm opt-out
  and `--check` reports it OK with that note (serve and query fall back to `en_core_web_sm`; builds need `[parser]`). A
  turn costs about 90 ms to parse on CPU; the optional `gpu` extra adds `cupy-cuda12x>=13`, and `MK_SPACY_GPU=1` makes
  the process ask spaCy for a GPU (`spacy.prefer_gpu()`, once, before its first model loads).
- **One loader, three policies.** Every spaCy reader in the kernel takes the process's one copy of a model (a process
  that held three trf copies, about 2.5 GB resident, holds one, about 1.2 GB; NER is off). Build and append fail loudly
  over a missing model. Serve's readers (`/context` identity lines, the `identity-lines` op, the tense reading behind
  relative-time rendering) fall back to `en_core_web_sm` with a warning and `parser_note("query")`, so `/context` is
  never empty over a missing model. A turn falls back as before.
- **Whitespace.** An empty or whitespace-only text is tokenized, not parsed (the trf model raised on it, so one blank
  utterance could kill a build or a plan). A turn is parsed stripped and returned over the original text, so
  `" hum a tune"` reads as a command like `"hum a tune"` (before, `hum` read as an interjection and the turn went
  unread). A lexical constraint check whose parse fails is `"unknown"`; it never drops the step.
- Details: [docs/runtime-contract.md](docs/runtime-contract.md#the-parser).

### Release hygiene

- **`release-check` workflow.** `.github/workflows/release-check.yml` (started by hand) runs the suite on the trf default
  parser, which CI skips; `docs/RELEASING.md` says to run it before tagging.
- **CI runs again.** `ci.yml` set the dead proxy as both `HTTPS_PROXY` and `https_proxy`; Actions reads env names without
  case, so the file failed to parse and no workflow ran on a push since 1.0. One spelling is kept.
- **Messages that say what happened.** A resume after a catalogue reload or a plan expiry says the plan expired or the
  catalogue was reloaded ("re-plan with committed"); a plan made during a reload says "runtime busy: reloading the tool
  catalogue". A call that waited behind a catalogue reload no longer starts an engine for the slot the reload dropped, and any failure of the
  registry on a catalogue is a refusal naming the tools, not a 500.
- **Docs.** The README's runtime-grammar section points at `docs/runtime-contract.md` instead of repeating it; the
  contract documents `plan_id`, `bonfire_id` and `turn`, recovery with `committed`, and the cold-start `503` of a
  catalogue `PUT`.

### Answer side

- **Removed:** `memory_kernel.comprehend` (its `_wh_slot`/`WH_LEMMAS` are replaced by `runtime.shape.read_shape`).
- **The wh word lists are retired.** What a question asks for is read from its parse (`runtime.shape.read_shape`, rules
  in `data/runtime-grammar/en/shapes.json`) in the answer directive, the context frame's shape and the query-semantics
  question words. The retired lists were `comprehend.service._wh_slot` / `WH_LEMMAS` (the `comprehend` package is gone),
  `serve.answer.WH` and `query_semantics._QUESTION_WORDS`. `serve` parses a question with the turn parser, not the
  extraction one.
- **What changes.** The directive of a plural wh phrase follows its morphology (a `list` directive for "Which dairy-free
  flavors does Kira Dace enjoy?", the entity directive for "What similar collectible do Milo and Ana own?"); "whose" asks
  for a person; a yes/no question that embeds a wh word ("Did Ana say who left?") gets no open-question directive; a
  degree ("how old") keeps the entity directive. A frame's `shape` is now a shape label (`what`, `list`, `how_many`,
  `polar`, ...), never a surface wh word, and a cached frame takes the shape read from the question.

### `mk serve` and the CLI

- **New routes.**
  - `POST /v1/bonfires/{id}/plan` and `POST /v1/bonfires/{id}/plan/{plan_id}/resume`.
  - `POST /v1/bonfires/{id}/teach` and `POST /v1/bonfires/{id}/teach/{draft_id}/confirm`.
  - `POST /v1/bonfires/{id}/plan/{plan_id}/retry` issues a refused step again (`Runtime.retry`).
  - `GET /v1/bonfires/{id}/constructions`.
  - `GET` and `PUT /v1/bonfires/{id}/catalogue`: the bonfire's tool catalogue. `PUT` validates the file, installs it
    atomically and reloads the bonfire's runtime (400 with the reason, the previous catalogue stays; 409 in store mode).
    `GET` returns `{sha256, tools}` with the tool rows, and so does `PUT`.
  - **One load gate.** `PUT` and `mk catalogue` refuse (400 / exit 2) a catalogue the runtime could not load: a tool named
    like a keyword, a name that cannot be a construction id, two names equal ignoring case, a card that does not compile.
    The previous catalogue stays.
- **A catalogue-card turn records its standing.** An attached catalogue card has no stored row; the turn's standing write
  skips it, as it skips base constructions, instead of failing for every construction in the turn.
- **`committed` on a plan request.** An optional list of `{tool, args, value}`: effects already done in the world
  (`Runtime.plan(committed=...)`). Step ids in HTTP are the `<plan_id>:<i>:<occ>` form.
- **A catalogue upload reloads the runtime.** In-memory plans are dropped (their ids answer 404), and the upload
  first waits for in-flight calls, up to `MK_SERVE_TEACH_WAIT_MS`. An engine that is still starting counts as a call in
  flight, so an upload during a cold start waits for it and never leaves a started engine outside the slot table.
- **Engines.**
  - One runtime per bonfire. Its engine starts on first use.
  - At most `MK_SERVE_MAX_RUNTIME_ENGINES` engines run (default 2), on ports from `MK_SERVE_RUNTIME_PORTS` (default
    `9712-9719`; a port something else holds is skipped). Past the cap, the least recently used idle engine is stopped
    first. An engine is never killed by another process while the process that started it lives.
  - Engines stop with the server.
- **Recall is cache-only.** An empty-result repair's fallback `recall` asks the bonfire's `/context` caches only, so a
  plan never makes a paid call; a miss is the diagnostic `recall unavailable: no cached context`.
- **While a bonfire teaches.** Its plans wait up to `MK_SERVE_TEACH_WAIT_MS` (15,000 ms), then answer `unread`
  ("runtime busy: teaching"). Its resumes and retries wait up to
  `MK_SERVE_RESUME_WAIT_S` (30 s), then answer 503 with `Retry-After`.
- **Store mode.** With `--store-uri`, teach and confirm answer 409 and no standing is recorded (1.4).
- **Bodies.** A string that is not valid UTF-8, or a field past its cap (text 4,000 characters, ids 256), is a 422.
- **`mk plan BONFIRE_DIR "text" --speaker NAME [--chat ID]`** prints one turn's `TurnPlan` JSON (for debugging). It
  never writes the bonfire (a read-only store, no standing).
- **`mk catalogue BONFIRE_DIR FILE`** validates a tool catalogue and installs it as the bonfire's
  `runtime/catalogue.json`; `mk plan` reads that catalogue.
- **`mk keywords`** prints the keyword glossary.
- **`/context` takes the reading's answer shape.** An optional `"answer_shape"` in the body (a `TurnPlan`'s
  `answer_shape`) is the frame's shape in place of the wh-word rule; the answer directive (`serve.answer.answer_prompt`)
  takes it the same way. Without one, nothing changes.

### Removed

- `memory_kernel.grammar.QUERY_RECIPE_LEXICAL_CUES` (a word list per query recipe) and the `recipe_cues` argument of
  `QuerySemanticsService.analyze_query`; `QueryAnalysis` no longer has `recipe_lexical_hits`. Nothing read them.
- The legacy grammar retrieval, unused by build, runtime and serve: `grammar/query_semantics.py`,
  `grammar/query_recipes.py`, `grammar/semantic_predicates.py` and `grammar/planner.py`, with their word lists, and
  their `memory_kernel.grammar` exports (`GrammarPlanner`, `QuerySemanticsService`, `QueryAnalysis`, `QueryProfile`,
  `get_query_semantics`, `build_grammar_query_plan`, `shared_construction_recipes`, `SemanticPredicateService`,
  `get_semantic_predicate_service`).
- `grammar/semantic_operators.py` is reduced to the tense and aspect reading `knowledge.temporal_deixis` uses. The
  profile holds `tense_aspects` only; the operator word lists and frame tables are gone, and a text with no parse has
  no tense (it is not guessed from words). The future auxiliaries ("will", "shall") are in
  `data/runtime-grammar/en/tense.json`. `construction.DefaultGrammarWorksetBuilder` (never instantiated) is deleted.
- The graph build reads a pronoun's person, a possessive's owner and a determiner's definiteness from the parse
  (`Person`, `Poss`, `Definite`, the PTB `PRP`/`PRP$` tags). Capture leaves carry no morphology, so
  `data/pronouns.json` gives person, number and gender per pronoun form (`grammar/pronouns.py`). The closed word sets in
  `construction/facet_resolver.py`, `construction/mapping_shadow.py`, `grammar/topic_grammar.py` and
  `grammar/triple/_dep_helpers.py` are gone, and `construction.referents` keeps only `classify_mention` (it takes a
  parsed span) and `MentionAnalysis`. "myself", "mine" and "yourself" as a whole role now bind to the speaker or addressee.
- WordNet noun lookups take the lemma they are given (WordNet's own morphology reads a plural); the suffix
  singulariser and the regex stemming fallback in `grammar/wordnet_semantics.py` are gone.
- The envelope renderer reads first- and second-person pronouns, their contractions ("I'm", "you've", "we'd") and
  plural address ("you two", "we all") from the parse instead of a pronoun pattern and a clitic table. The auxiliary
  forms are in `data/pronouns.json`. Text the old pattern corrupted ("I`m" -> "Milo`m") is left as written, and
  "took us two hours" keeps "two". A pronoun the tagger fuses with punctuation ("--I'm") is no longer rewritten.
  The renderer now parses every non-empty utterance, so it needs the extraction spaCy model and fails loudly without it, as
  extraction does.
- The jev2 shape filter's locative prepositions ("in", "at", "near", ...) are in `data/runtime-grammar/en/locative.json`;
  replayed over the ten recorded conversations' recorded alignments, the filter keeps and drops exactly what it did.
- The closed-class anchor forms (the function words a phrasal card may not be anchored on, and that the provisional-unit
  shim fences) are data in `data/closed-class-anchors.json`: one set for every parser, since the anchors are a property of
  English. The set is unchanged (the shim text, and so the capture cache keys, are byte-identical).
  `scripts/derive_closed_class_anchors.py` derives a set for any spaCy model as evidence for reviewing it.

## 1.3.1 (2026-10-18)

- The answer context's entity sheet keeps an entity's identity lines first: copular statements about the entity
  itself ("Kira Dace is a security researcher at Zorvex"), found per line from the dependency parse (a copular
  root with a nominal predicate whose subject is the entity's own name, or "I" when the entity is the speaker).
  At most `identity_lines_cap` (default 3) are pinned; the most recent lines fill the rest of `sheet_lines_cap`,
  so an entity with many statements on one date no longer loses who it is. A sheet with no identity line is
  unchanged.

## 1.3.0 (2026-10-17)

### Appends and the forward chain

- Forward-chain append: `mk append` and `append_session` take ONE session file, keys are frozen at first mint, name claims add `SAME_AS` links and route forward, the write only adds (orphan/drift reports and `--prune-orphans` are gone), search follows `SAME_AS`, and an append's parse, Jev and embedding work no longer grows with history (a few bookkeeping costs still do: see the README's **Time**). A bonfire in format v1 must be rebuilt.
- A one-message session passes the forward gates (zero picks, advisory `ctx` and `jev1` all-labels shares, Purpose-only C2); a bonfire's bootstrap is built without `--strict-gates`. A session written but not committed must be rerun before any other (`PendingSession`). A timestamp inversion is reported in the run entry and the append result.
- An append whose question coding lands on a line it appends to a stored entity's summary passes the C2 `codings_anchored` invariant (it read only the entity's own summary lines and refused the append, which then blocked every later session of the bonfire).
- An append's `sanity` stage writes a receipt (`mk-append-receipt/v1`: gate results, mints, reuses, routes, links, conflicts and the referents decisions) instead of searching every taxonomy question; it calls no provider. A full build's `sanity` is unchanged.
- Bonfires: `mk bonfire init` and `mk append` grow one conversation session by session (one persistent shard with a fixed budget, a per-message capture cache), with optional `session_N_conversation` chat keys (one boundary walk per chat; single-chat builds unchanged).
- Library API (`memory_kernel.api.append_session`, `context`) and `mk serve` (the `serve` extra): a warm per-bonfire HTTP context server that swaps to a new run (in the background) only once it is sealed, with query-time Jev calls on a separate query shard and a query cache per bonfire; a failing query frame degrades an answer instead of failing it.
- Bonfire storage in MongoDB (the `store` extra): content-addressed GridFS blobs, versioned manifests and a per-bonfire lease; `mk store push` / `pull`, `append_session(store=...)` and `context(store=...)`, `mk serve --store-uri`, run retention (`mk bonfire prune`); a plain `mk append` refuses a store working copy.

### Ingestion and boundaries

- Input: optional `conversation["agent_handles"]`, and per turn `addressees` and `timestamp` (ISO with timezone); the number of sessions is unbounded.
- An agent mention anywhere in a turn makes the agent an addressee; at the start or end it is stripped before parsing, in the middle it becomes the bare name.
- Message cleanup before parsing: links become `[link: host]`, data URIs `[embedded web page]`, HTML its visible text, runs of 80+ characters `[blob]`.
- Long messages are split into parts `<id>#1..#n` (at most 1,900 characters and 20 sentences or lines); an image caption stays on part 1.
- "You" resolves only through a message's `addressees` when the conversation carries any addressing; otherwise the two-party rule applies. Messages order by source order, not by id digits.
- `mk build --boundary {walk,segmenter}` chooses how a bonfire's episodes are cut. `walk` (the default) is the Jev
  topic walk and is unchanged: the boundary stage's outputs and `build-config.json` are byte-identical to before this change; Sagas gain centroids (next bullet). `segmenter` is for talks
  and documents: embedding cuts per session (target 60 messages, ceiling 150), no Jev request, Voyage spend on the
  `boundary` cost row; an append to a `segmenter` bonfire adds new Sagas and continues the numbering.
- Every Saga stores `centroid_embedding` (the L2-normalised mean of its messages' voyage-3 vectors),
  `centroid_model` and `centroid_messages`, under both policies. The write embeds only texts missing from the cache.
  The centroids are inert: nothing reads them yet.

### Referents and kinds

- `mk build --referents {phrase,structure}` (default `structure`) chooses how entities are keyed. A new build is
  `structure`; a config with no `referent_policy` reads as `phrase`, and a `phrase` build writes no such key, so every
  existing bonfire stays `phrase` and an append inherits its value. Re-keying an existing bonfire is not done.
  `structure` needs WordNet (NLTK data) and is refused without it. With `phrase`, the projection's rows, forward
  chain and `build-config.json` are identical to before the option existed, checked on a real bootstrap projection.
- Mentions (`structure`): a filler is cut into referring cores. A nested frame's tokens are cut out, conjuncts are
  split ("Kira and Milo" is two referents), and a name, or possessor + compounds + head, gives the key. A 3rd-person
  pronoun possessor is dropped like a determiner ("their phone" is "phone"); a nominal possessor keeps its own key
  without determiners ("the Brask Labs' device" and "Brask Labs' device" are one key); "my phone" is qualified by its
  owner as before. A name compound of a common noun ("the Zorvex spyware") is its own referent keyed "zorvex spyware",
  and the name inside it (Zorvex) is a mention of that message; it is not re-keyed to the name. A lowercased name the
  conversation knows ("the brask labs") is a name. A partitive ("this kind of X") gives X when its head's first noun
  sense is `noun.quantity`, or is `kind.n.01` or a WordNet hyponym of it ("sort", "type"); a name compound on it ("a
  Zorvex kind of X") is an inner name; other cognition nouns ("the investigation of X") keep their own referent. Inner
  names are mentions only, never separate facts. A bare NOUN head whose first noun sense is `noun.quantity` and that
  has no complement dependents ("a lot") refers to nothing. A capture span that does not line up with the
  projection's parse (chat noise such as emoji or URLs shifts token indices) is realigned onto the exact token run of
  the filler text in the unit (the same characters, whitespace and tokenisation aside: "re-exploit" matches spaCy's
  "re - exploit"), the one nearest the given span. Only a filler whose text is not in the unit is parsed
  on its own; there a mention needs a nominal head (NOUN, PROPN, PRON) or a name, so a paraphrased clause ("until the
  conditions force a choice") keys no "force". The stats count both (`referents_realigned`,
  `referents_nonnominal_dropped`).
- Entity names change under `structure` (one entity per referent: "Kira" and "Milo", not "Kira and Milo"), and the
  message search keys hold entity names, so a `structure` bonfire's search keys differ from a `phrase` build of the
  same conversation.
- Kinds (`structure`): one kind per referent, decided over all its mentions: Person (speaker, a roster or deixis
  person, or name join), else a kind noun the text supplies (a predicative noun, said of the PPT role only; an
  appositive; a common noun before a name head, "the journalist Milo Dace"), else its own head-noun class ("the Zorvex
  spyware" is an Artifact; Zorvex gets nothing from it), else Place (only with a location-type role and a
  location-compatible class). Only a NOUN head has a class: an adjective predicate ("Kira is rich") supplies no kind.
  WordNet lexnames map to Person, Organisation, Artifact and Place; Organisation and Artifact are new labels.
  `noun.group` gives Organisation only when the noun's first sense is `organization.n.01` or a hyponym of it ("team",
  "company"); other group nouns ("meeting", "audience", "generation") give no kind. A
  WordNet Person never joins a roster. Evidence is kept in `chain/kinds.jsonl`; an append recomputes a stored entity's
  kind and replaces a stale kind label. `phrase` labels stay frozen.
- Events (`structure`): an Event is named by its frame ("own.01: arg0 Kira and Milo; arg1 Brask Labs"); the sentence
  moves to `source_text`, which answer context shows when present, on timeline and support lines alike. The Event's
  `key_text` is unchanged, so frame names alone leave the search keys as they were.
- Referent resolution in appends: a new `referents` stage decides specific common-noun mentions ("the kiln"). A stored entity is reused only when the people talking share it (same chat, or the speaker or a human addressee took part) and it was last mentioned in this chat within 3 days, or when Jev judges it the same thing; otherwise the mention is a new entity, always under a discriminated key (`kiln#1a2b3c4d`): the plain key names the kind, and plain entities from before this version stay candidates (`legacy` in `ground.jsonl`). Named, bare (kind) and pronoun mentions resolve as before. The chain gains `ground.jsonl` and `evidence.jsonl`. A bonfire made before the referents stage existed needs `mk bonfire ground <dir>` once ($0); `mk append` refuses without it.
- Alias join: a person named two ways in the conversation ("X is Y", "my name is X", "call me X", "X and Y are the same person") becomes one Person entity with `identity_aliases`; a question naming an alias finds that person's sheet; `scripts/identity_joins.py` prints claims, joins and conflicts of a built `graph.json`.

### Jev and embeddings

- The `jev2` candidate shortlist embeds with Voyage voyage-3 (questions as `query`, event renderings as `document`)
  instead of a local MiniLM on the CPU, which took about 25 minutes per pass on a chat of 1,800 messages. Each distinct
  text is embedded once and kept in a cache keyed by model, input type and text (`logs/09-jev2-voyage-cache.json`, an
  append's in the bonfire's `chain/embeddings/`), so a resumed or re-run stage pays nothing twice and gets the same
  shortlist. A build costs a few cents more, recorded as a `voyage` line on the stage's `<dir>-shortlist` cost-ledger
  row; `--cap-usd` stops it after the Voyage request that crosses it. `jev2` now needs `VOYAGE_API_KEY`: a pending
  `jev2` without it is refused at preflight. The shortlist (and so `jev2`'s outputs) differ from those of earlier
  builds, including for the later sessions of an existing bonfire. MiniLM is no longer needed: the `minilm` setup
  component, its pin and `--adopt-minilm-cache` are gone (`mk setup --component minilm` is an unknown component; a
  receipt that lists it is still read, and its `hf/` dir can be deleted), and the `build` extra no longer installs
  `sentence-transformers` or `huggingface_hub`.
- A jev2 stage started before 1.3.0 must be finished (with the code that started it) or its directory deleted before
  resuming: `jev2` records its shortlist (`09-jev2-001/shortlist.json`: the embedder and the shortlists' sha) and
  refuses, before saving its plan or sending any Jev request, a stage dir whose `alignments.json` came from other
  shortlists. A plan-only `jev2` call reads the Voyage cache and sends nothing. A new bonfire's `chain/embeddings/` is
  seeded with the bootstrap's jev2 cache as well as its events cache. A Voyage request of the shortlist carries at most
  64 texts and about 60,000 estimated tokens.
- A Jev choice off the top probability is corrected to the top option and recorded; the plan's wire-size check is advisory (a really oversized request is still refused).
- A taxonomy call has a 180 s deadline and is retried once.

### Capture and the warm slot

- `mk build --capture-shards N` (default 1, at most 8) runs the capture stage on N engine shards at the same time, behind one
  shared parser. N is clamped to the CPUs the process may use. A shard that stops early or crashes leaves no complete
  report, and the merge refuses an incomplete set. A capture keeps its shard count for life.
- Capture output no longer depends on job order: the engine's id counters are reset before every capture job.
- Warm capture slot (opt-in): `mk slot serve --socket PATH --data-dir DIR` keeps one armed FCG engine and the managed parser up, and a capture whose config names the socket (`capture_slot`) leases them instead of booting its own (a parser start and a grammar image load, twice per capture). One lease at a time: the slot is the capture mutex among the captures that name its socket, and a second such capture waits for it (15 min, then a retriable `slot busy`). Under a warm lease the capture runs one shard on the slot's engine, re-checks the leased processes in `/proc`, and ends each attempt with a release receipt that the merge gate checks as it checks a teardown. The slot grants warm only when its engine, parser and code pins, its engine port, its parser's data dir and the arming environment (`MK_FCG_*` shim settings) equal the capture's; otherwise it frees the ports and the capture runs cold under the lease. Resting the parser after a release and a boot run under deadlines (30 s, 180 s): an engine that stops answering is torn down, never waited on. `mk slot serve` listens before it clears orphans and boots, so a capture arriving meanwhile waits for the lease. A capture already complete takes no lease; no slot answering runs the capture cold, as before, and the result says so. `mk slot status` prints identities, served counts, RSS and the state with the time it was entered (`state_since`); `mk slot stop` tears down after the current lease. A slot that dies alone is cleaned up by the next one, which kills only the processes its state file proves it started and does not count a zombie its parent never reaped. With `capture_slot` unset nothing changes.

### Costs, caps and gates

- Stage caps, the boundary cap and the round ledger cap scale with messages / 369 above 369 messages; the 1.0.0 values hold up to 369. The `--cap-usd` cap still bounds a round.
- Stage caps are recorded, not enforced: a stage whose plan, estimate or spend passes its cap (the boundary, picks,
  referents, taxonomy, Jev, question, events, ctx, remine and write-stage Voyage caps) records `stage_cap_exceeded` with
  the cap and the spend or estimate in its plan, summary or result, logs a `stage-cap-exceeded` event, and goes on; the
  driver's per-conversation limit is recorded the same way (`conv_limit_exceeded`). `--cap-usd` is the one spend stop
  (see the README's **Limits**); remine now stops at it, resumably, where it kept a partial inventory. A boundary window
  whose request would pass the 31,744-byte wire limit keeps its most recent messages that fit (`window_trimmed` in the
  boundary summary; the plan keeps its figures) instead of being refused when sent; a window within the limit is
  unchanged; an arrival that cannot fit at all opens a new episode (`capacity_cuts`) instead of stopping the build.
- New `--max-shift-share` (default 0.10) and `--strict-gates`: the advisory gates (`jev2-aligned`, the boundary stop rules) stop the build only when strict.
- The `ctx-prior-fallback` gate is advisory in a full build too: a miss is recorded and stops the build only with `--strict-gates`. A document corpus of technical prose, mostly states and general statements, missed it at 0.41 against 0.40.
- labelq's gate no longer requires context from a minimum number of sessions; a corpus with few sessions passes on
  question quality alone (reject share and exact duplicates). The session counts are still recorded in the gate values.
- The integrated spend cap of a new shard is $25 (was $10), recorded in its sealed manifest as `integrated_cap_usd`: a conversation can have up to 4,612 messages before the round cap is refused (the source step's 4,096-message limit still applies first). A manifest without the field enforces $10. Because `ledger.py` changed, **shards made by 1.0.0 refuse to open ("round implementation changed") and must be rebuilt.**
- **Round directories made by 1.0.0 cannot be resumed by 1.3 (the round pins `ledger.py`). Finish or discard in-flight 1.0.0 builds first.**

### Build driver and tooling

- Stages run inside the driving process instead of one `mk _stage` child each (`capture` stays a child: it imports `pyfcg` and installs a signal handler). `MK_STAGE_CHILDREN=1` forces a child for every stage of a whole build, as before. `mk _stage` by hand is unchanged.
- A stage runs once, live: its own plan is recomputed and saved inside that call, and its checks and spend caps still run before its first paid call. There is no `plan` event and no `logs/driver/<stage>-plan.json` any more.
  There is no `logs/mk/imports-<stage>-plan.json` either: a stage runs once, so it has only the `-live` import audit.
- A build whose config holds a relative path (any path field, a frame dir, or `MK_STOPWORDS_FILE`) is refused at preflight, before any stage: a stage in the driver's process would resolve it against that process's working directory. `mk build` and `mk append` already pass absolute paths; library callers passing `overrides` must too.
- `scripts/check_input_noop.py` proves the input layer is a no-op for inputs without the new fields, against an extracted 1.0.0 tree.

## 1.0.0

First public release.

- `mk build`: builds a typed memory graph for one conversation in Neo4j through a fixed, resumable sequence of stages
  (`select` to `seal`), with spend caps, a cost ledger, a fail-closed ledger shard and a code census (the hashes of the package's files) written once at the start of a build.
- `mk ask`: answers a question from a built group; read-only on Neo4j; serves from its cache unless `--allow-paid`.
- `mk inspect`: read-only per-stage status of a conversation root.
- `mk setup`: fetches or adopts and verifies the pinned runtime data (FCG Go engine and grammar image, PropBank frames, the
  spaCy pipeline, WordNet, the tiktoken file, MiniLM, the parser model); `--check` is offline and read-only.
- The syntactic parser is part of the package (`memory_kernel.parser_service`, extra `parser`); `mk build` starts it on a loopback
  port and refuses a non-local parser unless `--allow-remote-parser`.
- Packaged data: the English stopword list, the remine unit price, the runtime pins, and a CC BY-SA 4.0 annex of 24 PropBank
  framesets.
- Extras: `build` (engine bridge and embeddings), `parser`, `dev`.
- `examples/`: a Neo4j compose file, an invented dialogue and a purpose file.
- Tuning knobs are read from `MK_<AREA>_<KNOB>` names only (boundary, spaCy parse cache,
  FCG, temporal deixis, stopwords file, spaCy model). The earlier private
  names are not read; a variable exported under an old name is silently ignored. The public data classes lost their
  historical prefix (`Fact`, `Topic`, `Episode`, `Hypergraph`, `Config`, `Message`, and so on).
