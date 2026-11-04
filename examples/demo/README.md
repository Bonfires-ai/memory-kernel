# The hosted demo

The page behind the demo: ask a bonfire built from `examples/synthetic_dialogue.json`, and see the answer with the messages
it came from.

`mk serve` has no authentication, so it stays out of this. `app.py` is the only thing the internet reaches. It has one
route that asks, on a read-only Neo4j driver, with a limit per address (`DEMO_PER_MINUTE`, default 6) and a daily cap on
paid questions (`DEMO_DAILY_QUESTIONS`, default 500). Answers are cached, so a repeated question is free.

To run it on one server with Docker, point your domain at the server, set it in `Caddyfile`, and put `NEO4J_PASSWORD`,
`OPENROUTER_API_KEY` and `VOYAGE_API_KEY` in `examples/demo/.env`. Then:

To try it on your own machine first, skip the domain and start only the app: `compose.local.yml` serves it on
<http://127.0.0.1:8081>, with no Caddy.

```bash
cd examples/demo
docker compose -f compose.yml -f compose.local.yml build demo
docker compose -f compose.yml -f compose.local.yml --profile seed run --rm seed
docker compose -f compose.yml -f compose.local.yml --profile seed rm -sf mongo
docker compose -f compose.yml -f compose.local.yml up -d demo
```

On a server:

```bash
cd examples/demo
docker compose build
docker compose --profile seed run --rm seed    # once; needs about 8 GB of memory, costs about one cent
docker compose --profile seed rm -sf mongo     # the seed's record is not needed to serve
docker compose up -d
```
