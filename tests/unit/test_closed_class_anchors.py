"""The closed-class anchor set is one data file: the capture shim built from it is unchanged."""

from __future__ import annotations

import hashlib

from memory_kernel.construction.fcg.provisional_units import provisional_units_shim_lisp

#: sha256 of ``provisional_units_shim_lisp(3)`` when the 187 forms were a Python frozenset (capture cache keys hash it).
SM_SHIM_SHA256 = "004f9aa884bc6c7bab6ec0981c02bea5385990904483f77b72b635be2f6fa413"


def test_the_shim_text_is_unchanged():
    assert hashlib.sha256(provisional_units_shim_lisp(3).encode()).hexdigest() == SM_SHIM_SHA256
