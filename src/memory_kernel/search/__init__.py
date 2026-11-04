"""Search policies and public search API contracts.

The backend-facing request/response normalizers that used to live in
``search.contracts`` were retired in the kernel v1.0 public-API pass: they had
zero consumers in kernel ``src``, zero in any host application's ``src``, and zero in
``scripts/`` — only the backend, which is non-binding and already broken against the kernel.

Callers that need a :class:`~memory_kernel.types.SearchRequest` construct it
directly.
"""

__all__: list[str] = []
