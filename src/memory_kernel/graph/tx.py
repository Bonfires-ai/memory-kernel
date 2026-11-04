"""One Neo4j transaction behind the async-driver interface.

``kg_writer``, ``kg_index`` and ``pipeline.graphdb`` take a driver and call ``async with driver.session() as s:`` then
``await s.run(query, **params)``. ``TxScope`` answers those calls on ONE open transaction, so a stage that takes a
driver can run inside a transaction that is committed (or rolled back) as a whole.

Runs are serial: none of those modules gathers queries. No DDL runs inside a transaction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class _TxSession:
    """The ``session()`` object: ``run`` goes to the transaction; leaving the ``async with`` closes nothing."""

    def __init__(self, tx: Any) -> None:
        self._tx = tx

    async def __aenter__(self) -> _TxSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def run(self, query: str, parameters: Any = None, **params: Any) -> Any:
        return await self._tx.run(query, parameters, **params)


class TxScope:
    """Quacks like the async driver for the modules above, over one transaction ``tx``."""

    def __init__(self, tx: Any) -> None:
        self._tx = tx

    def session(self, *args: Any, **kwargs: Any) -> _TxSession:
        return _TxSession(self._tx)

    async def close(self) -> None:
        return None


@asynccontextmanager
async def transaction(driver: Any, *, commit: bool) -> AsyncIterator[TxScope]:
    """Open one transaction on ``driver``. ``commit=False`` always rolls back; ``commit=True`` commits on a clean exit
    and rolls back and re-raises on an exception."""
    session = driver.session()
    try:
        tx = await session.begin_transaction()
        try:
            yield TxScope(tx)
        except BaseException:
            await tx.rollback()
            raise
        if commit:
            await tx.commit()
        else:
            await tx.rollback()
    finally:
        await session.close()
