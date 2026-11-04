"""PropBank frameset lexicon loader for the GitHub propbank-frames repo."""

from __future__ import annotations

import functools
import logging
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from memory_kernel.normalization import clean

logger = logging.getLogger(__name__)

PROPBANK_FRAMESET_REPOSITORY = "https://github.com/propbank/propbank-frames"
PROPBANK_FRAMESET_PATH_ENV = "PROPBANK_FRAMESET_DIR"


class PropBankFramesetError(RuntimeError):
    """Raised when the configured PropBank frameset lexicon cannot be loaded."""


@dataclass(frozen=True, slots=True)
class PropBankRoleDefinition:
    role: str
    description: str = ""
    function_tag: str = ""


@dataclass(frozen=True, slots=True)
class PropBankRolesetDefinition:
    roleset_id: str
    lemma: str
    name: str = ""
    aliases: tuple[str, ...] = ()
    roles: tuple[PropBankRoleDefinition, ...] = ()
    source_path: str = ""
    typed_aliases: tuple[tuple[str, str], ...] = ()


def propbank_frameset_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a local checkout of github.com/propbank/propbank-frames."""

    raw = str(path or os.getenv(PROPBANK_FRAMESET_PATH_ENV) or "").strip()
    if not raw:
        raise PropBankFramesetError(
            f"{PROPBANK_FRAMESET_PATH_ENV} must point to a local checkout of {PROPBANK_FRAMESET_REPOSITORY}"
        )
    root = Path(raw).expanduser().resolve()
    if (root / "frames").is_dir():
        return root / "frames"
    if root.is_dir() and any(root.glob("*.xml")):
        return root
    raise PropBankFramesetError(f"PropBank frameset directory has no frames XML files: {root}")


def load_propbank_rolesets(path: str | os.PathLike[str] | None = None) -> tuple[PropBankRolesetDefinition, ...]:
    """Load all roleset definitions from a local propbank-frames checkout."""

    frames_dir = propbank_frameset_root(path)
    rolesets: list[PropBankRolesetDefinition] = []
    for xml_path in sorted(frames_dir.glob("*.xml")):
        rolesets.extend(_rolesets_from_file(xml_path))
    if not rolesets:
        raise PropBankFramesetError(f"PropBank frameset directory contained no rolesets: {frames_dir}")
    return tuple(rolesets)


def load_propbank_roleset_index(path: str | os.PathLike[str] | None = None) -> dict[str, PropBankRolesetDefinition]:
    """Return roleset definitions keyed by roleset id -- CACHED per resolved frames directory.

    Parsing the ~7,566-file propbank-frames checkout costs 0.7-0.8s (measured, this host) --
    dominated by XML parsing, not I/O. Every caller that only *reads* the index (the segment
    projection path chief among them: ``image_frames_with_outcomes`` calls this once per call,
    and the exam path calls that once per segment -- ~117 calls/exam, the whole 206s wall,
    an earlier finding) was re-paying that cost on every call for an index that never
    changes mid-run. The cache below removes the re-parse; identity of the returned object is
    preserved across cache hits, which is safe because no caller mutates it (audited 2026-09-06:
    every production and script call site only reads via ``[id]``/``.get``/``in``/iteration --
    grep for ``roleset_index.(pop|update|clear|setdefault)`` or subscript-assignment across the
    repo returns nothing).

    CACHE KEY: ``(str(resolved_frames_dir), frames_dir.stat().st_mtime_ns)``. The directory's own
    mtime -- not a per-file manifest digest -- is the version signal: reading it is a single
    `stat()` call (microseconds) versus the ~50-80ms a full `glob()` + per-file `stat()` manifest
    costs on this checkout, which would blow the "cached call costs <5% of the first" bar this
    unit is held to. Directory mtime changes whenever an entry is added, removed, or renamed
    inside `frames_dir` -- the shape every real update takes here: a fresh git checkout, a new
    roleset-annex drop, a replaced release tree. It does NOT change from an in-place content edit
    of an existing filename that keeps its own path (rewrite-without-rename) -- not a pattern this
    loader's callers use, but the true worst case, so :func:`clear_propbank_roleset_index_cache`
    is provided as an explicit escape hatch for anyone who needs to force a reload regardless.
    """

    frames_dir = propbank_frameset_root(path)
    signature = frames_dir.stat().st_mtime_ns
    return _cached_roleset_index(str(frames_dir), signature)


@functools.cache
def _cached_roleset_index(frames_dir: str, signature: int) -> dict[str, PropBankRolesetDefinition]:
    """The memoized parse. Split from :func:`load_propbank_roleset_index` so the cheap resolve
    (path -> frames_dir -> mtime) always runs, and only the expensive parse is memoized."""

    del signature  # part of the cache key only -- @functools.cache keys on every argument
    return {roleset.roleset_id: roleset for roleset in load_propbank_rolesets(frames_dir)}


def _rolesets_from_file(path: Path) -> list[PropBankRolesetDefinition]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        recovered = _recover_rolesets_from_malformed_xml(path)
        if recovered:
            return recovered
        raise PropBankFramesetError(f"PropBank frameset XML is invalid: {path}") from exc
    out: list[PropBankRolesetDefinition] = []
    for predicate in root.findall(".//predicate"):
        lemma = _lemma(predicate.get("lemma") or path.stem)
        for roleset in predicate.findall("roleset"):
            roleset_id = clean(roleset.get("id"))
            if not roleset_id:
                continue
            out.append(
                PropBankRolesetDefinition(
                    roleset_id=roleset_id,
                    lemma=lemma or _lemma(roleset_id.split(".", 1)[0]),
                    name=clean(roleset.get("name")),
                    aliases=tuple(_aliases(predicate, roleset)),
                    roles=tuple(_roles(roleset)),
                    source_path=str(path),
                    typed_aliases=_typed_aliases(predicate, roleset),
                )
            )
    return out


def _recover_rolesets_from_malformed_xml(path: Path) -> list[PropBankRolesetDefinition]:
    """Recover roleset schema from PropBank files with malformed examples."""

    parser = _RolesetSchemaParser(source_path=path)
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.rolesets


class _RolesetSchemaParser(HTMLParser):
    def __init__(self, *, source_path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.source_path = source_path
        self.rolesets: list[PropBankRolesetDefinition] = []
        self._predicate_lemma = ""
        self._predicate_aliases: list[str] = []
        self._predicate_typed_aliases: list[tuple[str, str]] = []
        self._roleset_id = ""
        self._roleset_name = ""
        self._roleset_aliases: list[str] = []
        self._roleset_typed_aliases: list[tuple[str, str]] = []
        self._roles: list[PropBankRoleDefinition] = []
        self._alias_target: str | None = None
        self._alias_chunks: list[str] = []
        self._alias_pos = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        if tag == "predicate":
            self._predicate_lemma = _lemma(attrs_map.get("lemma") or self.source_path.stem)
            self._predicate_aliases = []
            self._predicate_typed_aliases = []
        elif tag == "roleset":
            self._roleset_id = clean(attrs_map.get("id"))
            self._roleset_name = clean(attrs_map.get("name"))
            self._roleset_aliases = []
            self._roleset_typed_aliases = []
            self._roles = []
        elif tag == "alias":
            self._alias_target = "roleset" if self._roleset_id else "predicate"
            self._alias_chunks = []
            self._alias_pos = clean(attrs_map.get("pos")).lower()
        elif tag == "role":
            self._append_role(attrs_map)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "role":
            self._append_role({key.lower(): value or "" for key, value in attrs})

    def handle_data(self, data: str) -> None:
        if self._alias_target is not None:
            self._alias_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "alias":
            value = clean("".join(self._alias_chunks))
            if value and self._alias_target == "roleset":
                self._roleset_aliases.append(value)
                if self._alias_pos:
                    self._roleset_typed_aliases.append((_lemma(value), self._alias_pos))
            elif value:
                self._predicate_aliases.append(value)
                if self._alias_pos:
                    self._predicate_typed_aliases.append((_lemma(value), self._alias_pos))
            self._alias_target = None
            self._alias_chunks = []
            self._alias_pos = ""
        elif tag == "roleset":
            if self._roleset_id:
                self.rolesets.append(
                    PropBankRolesetDefinition(
                        roleset_id=self._roleset_id,
                        lemma=self._predicate_lemma or _lemma(self._roleset_id.split(".", 1)[0]),
                        name=self._roleset_name,
                        aliases=tuple(_dedupe([*self._predicate_aliases, *self._roleset_aliases])),
                        roles=tuple(self._roles),
                        source_path=str(self.source_path),
                        typed_aliases=tuple(
                            dict.fromkeys([*self._predicate_typed_aliases, *self._roleset_typed_aliases])
                        ),
                    )
                )
            self._roleset_id = ""
            self._roleset_name = ""
            self._roleset_aliases = []
            self._roleset_typed_aliases = []
            self._roles = []
        elif tag == "predicate":
            self._predicate_lemma = ""
            self._predicate_aliases = []
            self._predicate_typed_aliases = []

    def _append_role(self, attrs: dict[str, str]) -> None:
        if not self._roleset_id:
            return
        role_id = clean(attrs.get("n"))
        if not role_id:
            return
        self._roles.append(
            PropBankRoleDefinition(
                role="arg" + role_id.lower().removeprefix("arg"),
                description=clean(attrs.get("descr")),
                function_tag=clean(attrs.get("f")),
            )
        )


def _aliases(predicate: ET.Element, roleset: ET.Element) -> Iterable[str]:
    values: list[str] = []
    for element in (*predicate.findall("alias"), *roleset.findall(".//alias")):
        value = clean(element.text)
        if value:
            values.append(value)
    return _dedupe(values)


def _typed_aliases(predicate: ET.Element, roleset: ET.Element) -> tuple[tuple[str, str], ...]:
    pairs = []
    for element in (*predicate.findall("alias"), *roleset.findall(".//alias")):
        lemma, pos = _lemma(element.text), clean(element.get("pos")).lower()
        if lemma and pos:
            pairs.append((lemma, pos))
    return tuple(dict.fromkeys(pairs))


def _roles(roleset: ET.Element) -> list[PropBankRoleDefinition]:
    roles: list[PropBankRoleDefinition] = []
    for role in roleset.findall("roles/role"):
        role_id = clean(role.get("n"))
        if not role_id:
            continue
        roles.append(
            PropBankRoleDefinition(
                role="arg" + role_id.lower().removeprefix("arg"),
                description=clean(role.get("descr")),
                function_tag=clean(role.get("f")),
            )
        )
    return roles


def _lemma(value: Any) -> str:
    return clean(value).lower()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


#: VerbNet release whose ``lexlink`` ids :func:`verbnet_classes` reports (3.4 is the newest in the checkout).
VERBNET_LEXLINK_VERSION = "verbnet3.4"


def verbnet_classes(roleset: str, path: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    """VerbNet class ids for one roleset, from its frame XML ``lexlinks`` (VerbNet 3.4 only).

    Document order, de-duplicated. Empty when the roleset has no 3.4 link or is unknown. Raises
    :class:`PropBankFramesetError` when no frames checkout is configured, like the roleset index.
    """

    frames_dir = propbank_frameset_root(path)
    signature = frames_dir.stat().st_mtime_ns
    return _cached_verbnet_index(str(frames_dir), signature).get(roleset, ())


@functools.cache
def _cached_verbnet_index(frames_dir: str, signature: int) -> dict[str, tuple[str, ...]]:
    """Roleset id -> VerbNet 3.4 class ids, parsed once per frames directory (same cache key as the index)."""

    del signature  # part of the cache key only
    out: dict[str, tuple[str, ...]] = {}
    for xml_path in sorted(Path(frames_dir).glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            logger.warning("verbnet lexlinks skipped for malformed frame XML: %s", xml_path)
            continue
        for roleset in root.iter("roleset"):
            roleset_id = clean(roleset.get("id"))
            if not roleset_id:
                continue
            classes = [
                clean(link.get("class"))
                for link in roleset.findall("lexlinks/lexlink")
                if link.get("resource") == "VerbNet"
                and link.get("version") == VERBNET_LEXLINK_VERSION
                and clean(link.get("class"))
            ]
            if classes:
                out[roleset_id] = tuple(dict.fromkeys(classes))
    return out


def verbnet_role_themes(roleset: str, path: str | os.PathLike[str] | None = None) -> dict[str, tuple[str, ...]]:
    """VerbNet thematic roles of each numbered role of one roleset, from its frame XML ``rolelinks``
    (``VERBNET_LEXLINK_VERSION`` only, the release :func:`verbnet_classes` reports): ``{"arg1": ("experiencer",)}``.

    Lowercased, document order, de-duplicated; a role with no such rolelink is absent; an unknown roleset is ``{}``.
    """

    frames_dir = propbank_frameset_root(path)
    signature = frames_dir.stat().st_mtime_ns
    return dict(_cached_role_theme_index(str(frames_dir), signature).get(roleset, {}))


@functools.cache
def _cached_role_theme_index(frames_dir: str, signature: int) -> dict[str, dict[str, tuple[str, ...]]]:
    """Roleset id -> {argN: VerbNet themes}, parsed once per frames directory (same cache key as the index)."""

    del signature  # part of the cache key only
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for xml_path in sorted(Path(frames_dir).glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            logger.warning("verbnet rolelinks skipped for malformed frame XML: %s", xml_path)
            continue
        for roleset in root.iter("roleset"):
            roleset_id = clean(roleset.get("id"))
            if not roleset_id:
                continue
            roles: dict[str, tuple[str, ...]] = {}
            for role in roleset.findall("roles/role"):
                n = clean(role.get("n")).lower()
                themes = _dedupe(
                    clean(link.text).lower()
                    for link in role.findall("rolelinks/rolelink")
                    if link.get("resource") == "VerbNet"
                    and link.get("version") == VERBNET_LEXLINK_VERSION
                    and clean(link.text)
                )
                if n and themes:
                    roles["arg" + n.removeprefix("arg")] = tuple(themes)
            if roles:
                out[roleset_id] = roles
    return out


__all__ = [
    "PROPBANK_FRAMESET_PATH_ENV",
    "PROPBANK_FRAMESET_REPOSITORY",
    "PropBankFramesetError",
    "PropBankRoleDefinition",
    "PropBankRolesetDefinition",
    "load_propbank_roleset_index",
    "load_propbank_rolesets",
    "propbank_frameset_root",
    "verbnet_classes",
    "verbnet_role_themes",
]
