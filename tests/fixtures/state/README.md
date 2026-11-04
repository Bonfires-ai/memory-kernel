# State fixtures

These fixtures were frozen from real 1.8 code at mk `d918b782` (on H18 `9fcb958d`) by a one-off freeze script:
`project` -> `from_bootstrap` / `advance` -> commit / load -> `view()`, with the surfaces blanked and the rolesets
added; the row fixtures come from the real `write_rows` full build, then an append, on a scratch Neo4j.

That code is deleted in 2.0, so the fixtures cannot be regenerated from this tree. Regenerate them from a 1.8 checkout.
