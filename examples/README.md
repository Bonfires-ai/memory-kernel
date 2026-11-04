# Examples

| file | what |
|---|---|
| `synthetic_dialogue.json` | an invented conversation between two people over three dated sessions, with six questions in `qa`. No real data. |
| `purpose.txt` | a general research question for `--purpose-file`. |
| `docker-compose.yml` | Neo4j 5.26 (community) on `localhost:7687` and MongoDB 7 on `localhost:27017`, the services a bonfire needs besides the providers. |

## 1. Look at the input

`synthetic_dialogue.json` is a sample object: `sample_id`, a `conversation` with `session_N` message lists (`speaker`, `dia_id`,
`text`) and a `session_N_date_time` for each, written `H:MM am|pm on D Month, YYYY`, and a `qa` list (`question`, `answer`,
`evidence`, `category`). The build never reads `qa` while it builds; `select` only counts it.

## 2. The first stage, free, needs no setup and no services

```bash
mk build examples/synthetic_dialogue.json --workdir work --group example-dialogue --purpose-file examples/purpose.txt --through select
```

It prints the selection (24 messages, 3 sessions, 6 questions) and writes `work/example-dialogue/01-select-001/`. No provider is
called and Neo4j is not touched. Then:

```bash
mk inspect work/example-dialogue
```

shows `select` as done and every other stage as pending.

## 3. Set up the runtime data

```bash
pip install ".[build,parser,serve,ops]"
mk setup
mk setup --check
```

The later stages need the engine, the frames, the models, the WordNet corpus and the parser (see the README). `mk setup --check`
lists everything that is missing; a build is refused at preflight, before any stage runs, only for what the stages it would
run need, so `mk setup --check` may exit 1 and a short build still run: `--through select` needs nothing, `--through boundary` needs the WordNet corpus and the grammar image (and an OpenRouter key
when it runs), `--through capture` adds the engine binary and the parser, and `--through events` the frames and a Voyage key or its cache (the stage embeds the messages); `--through build_dry`, the
default, needs nothing more. `--through write` adds an embedding source (the Voyage key or `--embedding-cache`, which `events` does not read) and Neo4j:
the password, a server that answers and an empty group.

## 4. Seed a bonfire

```bash
export NEO4J_PASSWORD='choose-a-password'
export MK_OPS_URI='mongodb://localhost:27017'
export OPENROUTER_API_KEY=...        # paid stages
export VOYAGE_API_KEY=...            # message, graph and query embeddings
docker compose -f examples/docker-compose.yml up -d --wait
mk build examples/synthetic_dialogue.json --workdir work --group example-dialogue --purpose-file examples/purpose.txt --seed
```

`--seed` runs every stage and then makes the group a bonfire. `--group` must equal the input's `sample_id`,
`example-dialogue` here. This takes about five minutes on a 16-thread machine, spends about one cent (the cap is
`--cap-usd`, default 20) and sends the example's text to the providers. Re-run the same command to resume.

## 5. Ask

```bash
mk ask example-dialogue "What is Tomas building from a kit?" --allow-paid
```

The answer is printed; add `--json` for the query frame and the retrieved context. The expected answers are in the `qa` list.

## 6. Serve

```bash
mk serve --host 127.0.0.1 --port 8740 &
curl -s localhost:8740/healthz
curl -s -X POST localhost:8740/v1/bonfires/example-dialogue/context \
    -H 'content-type: application/json' -d '{"query": "What is Tomas building from a kit?"}'
```

`/context` returns the evidence, not an answer: the context block, the ids of the messages it came from, and the
question's frame. `mk serve` has no authentication, so keep it on `127.0.0.1`.
