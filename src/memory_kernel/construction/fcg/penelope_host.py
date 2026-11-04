"""Repoint FCG-Go's own syntactic-analysis fetch at our self-hosted parser service.

THE DEFECT (confirmed empirically 2026-08-27/28). FCG-Go's Lisp image ships
a ``*penelope-host*`` special variable that defaults to a remote, third-party research service
(``penelope-interface.lisp``, VUB AI Lab / Sony CSL Paris, 2017). ``:de-render-
constituents-dependents`` -- the de-render mode ``pyfcg.propbank.PropBankAgent`` and
``grammar_store.HASHED_LEMMA_GRAMMAR_CONFIG`` both use -- calls ``nlp-tools:get-
penelope-syntactic-analysis``, which POSTs through ``send-request`` to
``*penelope-host* + "/spacy-api/syntactic-parser"``. Every comprehend on that leg
therefore opened a live HTTPS connection to a third-party host (Vrije Universiteit
Brussel), measured at ~715ms mean -- an undeclared third-party dependency in the
comprehension critical path, and a data-egress path for a product whose purpose is
holding private conversations. Nowhere in this deployment was ``*penelope-host*``
ever overridden.

The stock capture path does NOT re-fetch: ``fcg-go::comprehend-and-extract-frames`` honours a supplied
``:syntactic-analysis`` -- its ``:de-render-constituents-dependents`` method builds the transient structure
from the analysis and calls ``nlp-tools:get-penelope-syntactic-analysis`` only on the nil branch (recorded
disassembly of the engine image; proven with a dead parser host
in a probe). The paths that DO re-fetch are the custom de-render modes
that do not forward the key: a pb-overlay de-render (not part of this package) and the card-lane
constituent shim (``provisional_units``). For those, repointing the host is the only mechanism.

Our self-hosted parser service (``memory_kernel.parser_service``) is a verified byte-identical drop-in:
30/30 captured parser fixtures match (modulo relational node-id renumbering), at
45.1ms vs 715.5ms -- about 16x faster. It serves the identical parser wire at
``/spacy-api/syntactic-parser`` on both the prefixed and bare paths.

``send-request``'s ONE piece of routing logic that matters here: a host containing
the literal substring ``"127.0.0.1"`` gets its route's leading ``/spacy-api`` or
``/embedding-api`` segment STRIPPED before the URL is built (a workaround for a
local dev server mounted at ``/``); every other host gets ``host + route`` verbatim.
There is no default host: the caller passes the parser it runs (``mk``'s
managed parser at ``http://127.0.0.1:<port>`` is served on both paths), or sets
:data:`PENELOPE_HOST_ENV`; with neither, arming refuses instead of leaving the
engine on its built-in remote host.

THE MECHANISM, following the shim precedent this module sits beside
(``qualified_hash.ANCHOR_HASH_SHIM_LISP`` / ``provisional_units.PROVISIONAL_UNITS_
SHIM_LISP``): ``pyfcg.routes.lisp_eval`` is the door. Unlike those two, this is not a
``defmethod`` install -- it is a ``setf`` on a special variable, so nothing here is
idempotent-by-construction; the guarantee instead comes from re-running it on every
``Engine.arm`` call (``pipeline.engine``), on the same code path that installs the
other shims, so the managed engine never starts without it running.

FAIL LOUD, NOT BEST-EFFORT. Like ``install_provisional_units_shim`` and unlike
``install_comprehend_cancel``: a failed install here would silently leave every
comprehend talking to a third party, which is precisely the defect this module
exists to close. :func:`install_penelope_host_shim` therefore RAISES when
``lisp_eval`` is unavailable, or when the readback FROM THE ENGINE does not equal
the value just set -- config that silently did not apply is the exact bug class the
readback exists to catch (config drift has bitten this codebase repeatedly). The one place this backs off
from a hard raise is host UNREACHABILITY at arm time (a TCP probe, not the ``setf``
itself) -- gated by :data:`PENELOPE_HOST_REQUIRE_REACHABLE_ENV`, default ON. The
raise-by-default choice matches the ``setf`` guarantee above: if the configured host
cannot be reached, every comprehend on this engine is about to fail anyway, and
failing at arm time is strictly better than paying that out per-statement. The
escape hatch exists only for topologies where DNS/network reachability genuinely
differs from the production compose network (bare-host scratch engines, CI) --
setting it OFF still logs the failure at ERROR, unmistakably, so the degraded state
is never silent.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any
from urllib.parse import urlsplit

from memory_kernel.config.flags import env_flag_unless_disabled

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PENELOPE_HOST",
    "PENELOPE_HOST_ENV",
    "PENELOPE_HOST_REQUIRE_REACHABLE_ENV",
    "configured_penelope_host",
    "install_penelope_host_shim",
]

#: THE NAME IS THE CONTRACT: the environment variable is read verbatim by name.
PENELOPE_HOST_ENV = "MK_FCG_PARSER_HOST"
#: Default-on escape hatch (`env_flag_unless_disabled` shape): OFF only degrades an
#: unreachable-at-arm-time host from a raise to a loud ERROR log. See module
#: docstring's "FAIL LOUD, NOT BEST-EFFORT" for why this is the one soft spot.
PENELOPE_HOST_REQUIRE_REACHABLE_ENV = "MK_FCG_PARSER_HOST_REQUIRE_REACHABLE"

#: No default host: a host is passed (``host=``) or configured
#: (:data:`PENELOPE_HOST_ENV`), else arming refuses. The engine image's own built-in
#: host is a remote public service, so "no host" must never mean "leave it there".
DEFAULT_PENELOPE_HOST: str | None = None

_REACHABILITY_TIMEOUT_S = 2.0


def configured_penelope_host() -> str | None:
    """The configured parser host, or :data:`DEFAULT_PENELOPE_HOST` (None: no host).

    Read LIVE (no import-time env reads) -- a deploy that exports
    :data:`PENELOPE_HOST_ENV` after import must still see it take effect on the
    next arm.
    """

    raw = str(os.environ.get(PENELOPE_HOST_ENV) or "").strip()
    return raw or DEFAULT_PENELOPE_HOST


def _lisp_string_literal(value: str) -> str:
    """A Lisp double-quoted string literal for ``value``. Escapes ``\\`` and ``"`` only."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _probe_reachable(host: str, *, timeout: float = _REACHABILITY_TIMEOUT_S) -> tuple[bool, str]:
    """Best-effort TCP reachability of ``host``'s ``scheme://hostname[:port]``.

    Not a real HTTP request -- a bare TCP connect is enough to distinguish "nothing
    is listening there" (the failure mode this exists to catch loudly) from an
    actual protocol error, which a real comprehend will surface on its own. Never
    raises: every failure mode collapses to ``(False, detail)``.
    """

    parsed = urlsplit(host)
    hostname = parsed.hostname
    if not hostname:
        return False, f"unparseable host {host!r} (no hostname)"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((hostname, port), timeout=timeout):
            return True, ""
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def install_penelope_host_shim(pyfcg: Any, *, host: str | None = None) -> str:
    """Point a live FCG-Go engine's ``nlp-tools::*penelope-host*`` at our service.

    Called from ``Engine.arm`` (``pipeline.engine``) beside the other shims, so the
    managed engine has it applied before the grammar image is loaded or any
    comprehend runs; ``Engine.set_parser`` calls it again to repoint the host.

    Returns the value read back FROM THE LIVE ENGINE (not the Python-side
    ``host``/``configured_penelope_host()`` value) -- callers that need to prove the
    setf actually took must assert against this return value or re-read the engine
    themselves, never against their own input.

    Raises ``RuntimeError`` when: ``lisp_eval`` is unavailable; the readback does
    not equal what was just set; or the configured host fails a TCP reachability
    probe at arm time AND :data:`PENELOPE_HOST_REQUIRE_REACHABLE_ENV` is not
    explicitly disabled (default: required). See the module docstring for why this
    is the one shim in this file that is NOT best-effort.
    """

    configured = host if host is not None else configured_penelope_host()
    if configured is None:
        raise RuntimeError(
            f"no parser host: pass host= or set {PENELOPE_HOST_ENV}; refusing to leave the engine on its built-in "
            "remote host"
        )
    target = configured.strip()
    if not target:
        raise RuntimeError(f"{PENELOPE_HOST_ENV} resolved to an empty host; refusing to arm a blank penelope-host")

    lisp_eval = getattr(getattr(pyfcg, "routes", None), "lisp_eval", None)
    if not callable(lisp_eval):
        raise RuntimeError(
            "pyfcg.routes.lisp_eval is required to repoint nlp-tools::*penelope-host*; without "
            "it every comprehend on :de-render-constituents-dependents keeps opening a live "
            "connection to the engine image's built-in remote host"
        )

    literal = _lisp_string_literal(target)
    raw = lisp_eval(f'(progn (setf nlp-tools::*penelope-host* {literal}) (format nil "~a" nlp-tools::*penelope-host*))')
    readback = str(raw or "").strip().strip('"')
    if readback != target:
        raise RuntimeError(
            f"nlp-tools::*penelope-host* readback {readback!r} does not match the configured "
            f"host {target!r} ({PENELOPE_HOST_ENV}) -- refusing to trust an engine that may "
            "still be pointed at its built-in remote host"
        )

    reachable, detail = _probe_reachable(target)
    if reachable:
        logger.info("FCG penelope-host armed: nlp-tools::*penelope-host* = %s (TCP reachable)", readback)
    else:
        message = (
            f"FCG PENELOPE-HOST UNREACHABLE AT ARM TIME: nlp-tools::*penelope-host* is set to "
            f"{readback!r} ({PENELOPE_HOST_ENV}) but a TCP probe failed ({detail}). Every "
            f"comprehend on :de-render-constituents-dependents will now ERROR rather than "
            "silently falling back to the built-in remote host -- the host is repointed "
            "unconditionally, not conditionally, so there is no fallback path to fall back to."
        )
        if env_flag_unless_disabled(PENELOPE_HOST_REQUIRE_REACHABLE_ENV, default=True):
            raise RuntimeError(message)
        logger.error(message)

    return readback
