"""Envelope-driven statement rendering — the kernel's ONE envelope renderer.

A statement minted from a chat turn is first-person and deictic ("I went there last
week"), which is a poor retrieval surface: the speaker is not in the text and "last
week" is not a date. The envelope (speaker, addressee, timestamp) carries exactly what
is needed to resolve those indexicals. Rendering alone gave a clear retrieval lift on
the development conversation (ranking untouched), but that lift was produced by an OFFLINE
script rewriting a corpus artifact, so a live bonfire got nothing from it. This package is
that renderer, moved in-tree.

## The contract for the producer (the embedding side)

```python
from memory_kernel.envelope import addressee_map, render, resolve_addressee

parties = addressee_map(corpus.values())            # once per corpus
row["utterance_rendered"] = render(                 # per row
    row["utterance"],                               # RAW text, never rewritten
    speaker=row["actor_id"],
    addressee=resolve_addressee(parties, row["session"], row["actor_id"]),
    ts=row["ts"],
)
```

`utterance` stays RAW and the statement hash stays `sha256(raw)[:16]` (the no-rewrite
law): the render never moves a hash, so artifact-to-artifact diffs stay
row-aligned. `utterance_rendered` is the derived field, and it is what the serve side
reads when the producer publishes rendered text.

## The serve side never renders

`search.recipe_primitives.served_text` returns `entry.rendered or entry.utterance`. A
row with an empty `utterance_rendered` is served RAW — it is never rendered on the fly.
That is not an optimisation: the existing corpus holds rows that were
ALREADY rendered upstream, and rendering those again would double-render them.
"""

from __future__ import annotations

from memory_kernel.envelope.render import render, resolve_temporal

__all__ = [
    "render",
    "resolve_temporal",
]
