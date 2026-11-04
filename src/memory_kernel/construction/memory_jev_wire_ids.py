"""Request-local aliases for opaque IDs; full identities remain in the journal."""

# One byte per token, with 1024 tokens left for provider framing.
MAX_WIRE_BYTES = 31744
