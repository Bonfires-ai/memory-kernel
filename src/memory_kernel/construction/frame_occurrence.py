"""Validated predicate occurrence identity, independent of modifiers and argument readings.

No module in this package imports it. It stays because out-of-tree harness code imports ``FrameOccurrence`` from here.
It is stdlib-only on purpose: pydantic is not a dependency of the package, and the field annotations never validated
anything (``dataclass`` ignores them), so ``__post_init__`` below is the whole contract.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameOccurrence:
    roleset: str
    source_tokens: tuple[str, ...]
    segment_index: int
    segment: str
    predicate_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.roleset, str) or not self.roleset.strip():
            raise ValueError("frame occurrence requires a roleset")
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("frame occurrence requires a nonnegative segment index")
        if (
            not isinstance(self.source_tokens, tuple)
            or not self.source_tokens
            or any(
                not isinstance(token, str) or not token or any(c.isspace() for c in token)
                for token in self.source_tokens
            )
        ):
            raise ValueError("frame occurrence requires exact nonspace source tokens")
        if (
            not isinstance(self.segment, str)
            or not self.segment.strip()
            or ("".join(self.segment.split()) != "".join(self.source_tokens))
        ):
            raise ValueError("frame occurrence segment does not match source tokens")
        if (
            not isinstance(self.predicate_indices, tuple)
            or not self.predicate_indices
            or any(
                type(index) is not int or index < 0 or index >= len(self.source_tokens)
                for index in self.predicate_indices
            )
            or tuple(sorted(set(self.predicate_indices))) != self.predicate_indices
        ):
            raise ValueError("frame occurrence requires ordered unique predicate coordinates")
