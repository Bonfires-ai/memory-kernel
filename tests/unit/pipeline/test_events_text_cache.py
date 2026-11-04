"""The boundary stage and the events stage embed the same string per message, through one cache class."""

from __future__ import annotations

from memory_kernel.ingest.episode_boundary import boundary_text
from memory_kernel.pipeline import embed
from memory_kernel.pipeline.stages import _events_core as ev


def _row(mid, order, speaker, text, captions=()):
    return {
        "message_id": mid,
        "order": order,
        "session_id": "s1",
        "speaker": speaker,
        "timestamp": "2026-01-01T00:00:00Z",
        "text": text,
        "attachments": [{"attachment_id": f"a{i}", "caption": c} for i, c in enumerate(captions)],
    }


def _capture_records(rows):
    """What capture + graph leave for these rows: one record per unit of every non-blank part, in part order.

    The units of a part tile its text (a sentence unit runs to the next sentence's first token), so two units
    stand in for a two-sentence message."""
    records = []
    for row in rows:
        parts = [row["text"], *(a["caption"] for a in row["attachments"])]
        unit = 0
        for part in (p for p in parts if p.strip()):
            cut = part.find(". ") + 2 if ". " in part else len(part)
            for piece in (part[:cut], part[cut:]) if cut < len(part) else (part,):
                records.append({**{k: row[k] for k in ("message_id", "session_id", "speaker", "timestamp")}})
                records[-1].update(unit=f"{row['order']:04d}/unit-{unit:02d}.json", text=piece)
                unit += 1
    return records


def test_boundary_and_events_embed_the_same_text():
    rows = [
        _row("m-1", 0, "Ada", "I moved to Lyon. It rains a lot.  "),
        _row("m-7#2", 1, "Bo", "  second half of a long post", captions=("a chart of rain",)),
        _row("m-9", 2, "Cy", "ok"),
    ]
    rows[2]["speaker"] = ""  # a message without a speaker: the text alone
    expected = {m["message_id"]: boundary_text(m) for m in ev.messages_of(_capture_records(rows))}
    assert embed.message_texts(rows) == expected
    assert expected["m-9"] == "ok" and expected["m-1"] == "Ada: I moved to Lyon. It rains a lot."
    assert expected["m-7#2"] == "Bo: second half of a long posta chart of rain"
