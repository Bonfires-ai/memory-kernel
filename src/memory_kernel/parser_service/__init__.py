# Derived from an internal parser service, file __init__.py, source sha256
# 9a1f75150fc21668324cc36460bb36f26417925baedd8d7bb9d56247bc4afc7e.
# Copied from the origin file; edits: this header, the package-relative imports and a prose-only scrub of docstrings and comments.
"""The standalone parser service behind the managed syntactic parser (``memory_kernel.parser_service``).

Origin scope (spaCy skeleton): tokenization, POS, dependency parse, sentence
segmentation, contraction/edge-punct normalization, and a facade compatible
with the FCG-Go parser client. Only the facade, the constituency layer and their
configuration are copied here; the origin service's other modules (its lemma
owner, native analysis routes, schemas) are not part of this package.
"""

from __future__ import annotations

__version__ = "0.2.0"

#: Bumped whenever the annotation-producing logic in this package changes in
#: a way that could shift downstream analyses (lemma tables, normalization
#: rules, segmentation heuristics, ...). Distinct from the spaCy model
#: version — this is OUR pipeline's version, stamped on every response
#: alongside {model, model_version} so bucket-keyed consumers can detect
#: drift instead of silently mismatching (lemma disagreement across consumers).
PIPELINE_VERSION = __version__

__all__ = ["PIPELINE_VERSION", "__version__"]
