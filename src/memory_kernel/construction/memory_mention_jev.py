"""Bounded Jev questions and receipt-bound reduction for licensed mentions.

This module prepares questions; it does not send requests or commit memory.
The host owns the provider ledger and archives the exact wire, raw, and receipt
bytes before passing their physical pins to an admission record.
"""

from __future__ import annotations

MAX_QUESTIONS = 64
MODEL = "typesafe/jev-1.13"
