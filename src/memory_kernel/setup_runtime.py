"""``mk setup``: fetch (or adopt), verify and place the pinned runtime data a build needs, and check it later.

Eight components, each pinned by sha256 in ``data/runtime-pins.json``:

===============  ==============================================================  ======================================
component        what                                                            lands in
===============  ==============================================================  ======================================
fcg-go           the FCG Go engine binary (fetched through pyfcg, never shipped)  ``<data>/fcg-go/<sha12>/FCG Go``
fcg-image        the grammar image the engine loads                              ``<data>/fcg-go/resources/<name>``
propbank-frames  upstream PropBank frame files at one commit                     ``<data>/propbank-frames/<commit12>/``
spacy-model      ``en_core_web_sm`` (a wheel URL, not a PyPI name)               the RUNNING interpreter's site-packages
spacy-model-trf  ``en_core_web_trf``, the transformer pipeline (a wheel URL)     the RUNNING interpreter's site-packages
wordnet          the NLTK WordNet corpus                                         ``<data>/nltk_data/corpora/wordnet``
tiktoken         the ``cl100k_base`` BPE file                                    ``<data>/tiktoken/<cache key>``
parser-model     the ``benepar_en3`` constituency model of the managed parser    ``<data>/nltk_data/models/benepar_en3``
===============  ==============================================================  ======================================

``<data>`` is ``--data-dir``, else ``MK_DATA_DIR``, else ``platformdirs.user_data_dir("memory-kernel")``. Receipts live in
``<data>/setup.json`` (what was placed, from where, with which digests, pinned or not); ``--check`` reads only that file
and the placed objects, never the network and never writes.

Rules this module keeps:

* Every object is fetched or copied into a FRESH staging dir under ``<data>/.staging`` and verified THERE; only a verified
  object is moved into place. A mismatch refuses and names the expected and the actual sha.
* Only the two engine components honour ``--allow-unpinned-engine``: the mismatching object is placed and the receipt
  records ``pinned: false`` with the actual shas. Nothing else has an override.
* pyfcg is called with a path that does not exist yet (``download_fcg_go`` deletes its target directory and
  ``load_resource`` skips a file that already exists without looking at it) and from inside the staging dir (the
  ``wget`` module it uses writes its temp file into the working directory). ``pyfcg.init(launch=True)`` is never called.
* ``zipfile`` drops the executable bit, so the engine binary is chmod-ed to the pinned mode before it is hashed.
* Linux x86_64 only: the pins exist for that platform alone.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.resources
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

PINS_SCHEMA = "mk-runtime-pins/v1"
RECEIPT_SCHEMA = "mk-setup/v1"
RECEIPT_NAME = "setup.json"
STAGING_DIR = ".staging"

COMPONENTS: tuple[str, ...] = (
    "fcg-go",
    "fcg-image",
    "propbank-frames",
    "spacy-model",
    "spacy-model-trf",
    "wordnet",
    "tiktoken",
    "parser-model",
)
#: The components ``--allow-unpinned-engine`` applies to.
ENGINE_COMPONENTS: tuple[str, ...] = ("fcg-go", "fcg-image")

OK, MISSING, MISMATCH = "ok", "missing", "mismatch"

#: ``component -> the --adopt-* option (dest name) that adopts it``; the CLI builds its flags from this table.
ADOPT_FLAGS: dict[str, str] = {
    "fcg-go": "adopt_fcg_go",
    "fcg-image": "adopt_fcg_image",
    "propbank-frames": "adopt_frames",
    "spacy-model": "adopt_spacy_model",
    "wordnet": "adopt_wordnet",
    "tiktoken": "adopt_tiktoken",
    "parser-model": "adopt_parser_model",
}


class SetupError(Exception):
    """A refusal (exit 2). The message names what was expected and what was found."""


# -- pins, data dir, receipts ------------------------------------------------------------------------------------------


def load_pins() -> dict[str, Any]:
    text = (importlib.resources.files("memory_kernel") / "data" / "runtime-pins.json").read_text()
    doc = json.loads(text)
    if doc.get("schema") != PINS_SCHEMA:
        raise SetupError(f"runtime-pins.json: schema {doc.get('schema')!r}, expected {PINS_SCHEMA!r}")
    return doc


def resolve_data_dir(explicit: str | os.PathLike[str] | None, environ: Mapping[str, str]) -> Path:
    """``--data-dir`` > ``MK_DATA_DIR`` > the platform user data dir. Lexical and absolute; nothing is created."""
    raw = explicit or environ.get("MK_DATA_DIR") or None
    if raw is None:
        import platformdirs

        raw = platformdirs.user_data_dir("memory-kernel")
    return Path(os.path.abspath(os.path.expanduser(raw)))


def read_receipt(data: Path) -> dict[str, Any]:
    """``<data>/setup.json``; an absent file is an empty receipt. Read-only."""
    path = data / RECEIPT_NAME
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {"schema": RECEIPT_SCHEMA, "components": {}}
    except OSError as exc:
        raise SetupError(f"{path}: {type(exc).__name__}: {exc}") from None
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise SetupError(f"{path} is not valid JSON ({exc}); delete it and run mk setup again") from None
    if not isinstance(doc, dict) or doc.get("schema") != RECEIPT_SCHEMA or not isinstance(doc.get("components"), dict):
        raise SetupError(f"{path}: not a {RECEIPT_SCHEMA} receipt; delete it and run mk setup again")
    return doc


def _write_receipt(data: Path, doc: Mapping[str, Any]) -> None:
    data.mkdir(parents=True, exist_ok=True)
    tmp = data / f".{RECEIPT_NAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, data / RECEIPT_NAME)


def managed_path(data: Path, component: str) -> Path | None:
    """The absolute path the receipt records for ``component`` when it exists on disk, else None (no verification)."""
    entry = read_receipt(data)["components"].get(component)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        return None
    path = data / entry["path"]
    return path if path.exists() else None


# -- hashing, downloading, staging ----------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(root: Path, *, suffix: str | None = None) -> tuple[str, int]:
    """sha256 of one line ``"<sha256 of file>  ./<relative path>\\n"`` per file under ``root``, sorted bytewise by relative
    path (``find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum``). ``suffix`` keeps only matching names;
    ``__pycache__`` dirs and ``.pyc`` files never count. Returns ``(digest, file count)``."""
    rels: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".pyc") or (suffix is not None and not name.endswith(suffix)):
                continue
            rels.append((Path(dirpath) / name).relative_to(root).as_posix())
    rels.sort(key=lambda r: r.encode())
    text = "".join(f"{sha256_file(root / rel)}  ./{rel}\n" for rel in rels)
    return hashlib.sha256(text.encode()).hexdigest(), len(rels)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def installed_version(dist: str) -> str | None:
    """The version of an installed distribution in the RUNNING interpreter, else None (a seam for the tests)."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mk_version() -> str:
    return installed_version("memory-kernel") or "unknown"


def download(url: str, dest: Path, *, timeout: float = 300.0) -> str:
    """Stream ``url`` into ``dest`` (a new file) and return the sha256 of the bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": f"memory-kernel-setup/{_mk_version()}"})
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SetupError(f"download of {url} failed: {type(exc).__name__}: {exc}") from None
    return h.hexdigest()


@contextlib.contextmanager
def staging(data: Path, name: str, *, keep_on_error: bool = False) -> Iterator[Path]:
    """A fresh dir under ``<data>/.staging``. Removed on exit; after a refusal it is kept (``keep_on_error``, for a
    download that cost bytes) so the fetched file can be adopted without fetching again."""
    root = data / STAGING_DIR
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=root))
    try:
        yield stage
    except SetupError as exc:
        if keep_on_error and any(stage.iterdir()):
            raise SetupError(f"{exc} (the fetched files stay in {stage})") from None
        shutil.rmtree(stage, ignore_errors=True)
        raise
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    else:
        shutil.rmtree(stage, ignore_errors=True)


@contextlib.contextmanager
def _pyfcg_context(directory: Path) -> Iterator[None]:
    """pyfcg prints to stdout (``--json`` must stay clean) and its ``wget`` writes a temp file into the working dir."""
    previous = Path.cwd()
    os.chdir(directory)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        os.chdir(previous)


def _safe_parts(name: str, what: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise SetupError(f"{what}: unsafe archive member name {name!r}")
    return path.parts


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


# -- the components ---------------------------------------------------------------------------------------------------------


@dataclass
class Context:
    data: Path
    allow_unpinned: bool = False
    verify: bool = False


@dataclass
class Row:
    name: str
    status: str = MISSING
    pinned: bool | None = None
    path: str | None = None
    detail: str = ""
    action: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "pinned": self.pinned,
            "path": self.path,
            "detail": self.detail,
            "action": self.action,
            "error": self.error,
        }


def _flat(doc: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in doc.items():
        if isinstance(value, Mapping):
            out.update(_flat(value, f"{prefix}{key}/"))
        else:
            out[f"{prefix}{key}"] = value
    return out


def parser_extra_installed() -> bool:
    """Whether the ``[parser]`` extra's packages are importable (found, not imported): trf cannot load without them."""
    return all(importlib.util.find_spec(n) is not None for n in ("spacy_curated_transformers", "torch", "transformers"))


class Component:
    """One pinned runtime object. ``digest`` describes a placed (or staged) object; ``expected`` is the pinned digest."""

    name = ""
    #: digest keys that are only known on some routes (the zip sha is unknown when a binary is adopted directly)
    optional: frozenset[str] = frozenset()

    def __init__(self, pin: Mapping[str, Any]) -> None:
        self.pin = pin

    @property
    def engine(self) -> bool:
        return self.name in ENGINE_COMPONENTS

    def skip_note(self) -> str | None:
        """Why this component is left out of this install (a note), or None when it applies."""
        return None

    def pin_id(self) -> str:
        return hashlib.sha256(json.dumps(self.pin, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    # -- the parts each component supplies
    def expected(self) -> dict[str, Any]:
        raise NotImplementedError

    def digest(self, obj: Path) -> dict[str, Any]:
        raise NotImplementedError

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        """Download into ``stage``: ``(staged object, digest facts known before hashing, origin url)``."""
        raise NotImplementedError

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        """Copy a local object into ``stage``: ``(staged object, digest facts known before hashing)``."""
        raise NotImplementedError

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        raise NotImplementedError

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        """A cheap presence check of a placed object; a problem text or None."""
        return None

    # -- shared behaviour
    def compare(self, digest: Mapping[str, Any]) -> list[str]:
        want, got = _flat(self.expected()), _flat(digest)
        problems = []
        for key, expected in want.items():
            if key not in got:
                if key not in self.optional:
                    problems.append(f"{self.name} {key}: expected {expected}, found nothing")
            elif got[key] != expected:
                problems.append(f"{self.name} {key}: expected {expected}, actual {got[key]}")
        return problems

    def check(self, ctx: Context, entry: Mapping[str, Any] | None) -> Row:
        row = Row(self.name)
        if entry is None:
            row.detail = f"not set up: run `mk setup --component {self.name}`"
            return row
        row.pinned = bool(entry.get("pinned"))
        path = ctx.data / str(entry.get("path", ""))
        row.path = str(path)
        if not entry.get("path") or not path.exists():
            row.detail = f"{path} is gone: run `mk setup --component {self.name}`"
            return row
        if row.pinned and entry.get("pin_id") != self.pin_id():
            row.status, row.detail = MISMATCH, "the pins changed since this was set up: run `mk setup`"
            return row
        problem = self.shape(path, entry)
        if problem:
            row.detail = problem
            return row
        if ctx.verify:
            digest = self.digest(path)
            recorded = _flat(entry.get("digests", {}))
            problems = self.compare(digest) if row.pinned else []
            problems += [
                f"{self.name} {key}: recorded {recorded[key]}, actual {value}"
                for key, value in _flat(digest).items()
                if key in recorded and recorded[key] != value
            ]
            if problems:
                row.status, row.detail = MISMATCH, "; ".join(problems)
                return row
        row.status = OK
        row.detail = "verified" if ctx.verify else "present"
        return row

    def install(self, ctx: Context, adopt: Path | None) -> dict[str, Any]:
        """Fetch or adopt, verify in staging, place, and return the receipt entry."""
        with staging(ctx.data, self.name, keep_on_error=adopt is None) as stage:
            if adopt is None:
                obj, facts, origin = self.fetch(stage)
                mode = "download"
            else:
                obj, facts = self.adopt(adopt, stage)
                origin, mode = str(adopt), "adopt"
            staged = self.digest(obj)
            digest = {**facts, **staged}
            problems = self.compare(digest)
            if problems and not (self.engine and ctx.allow_unpinned):
                raise SetupError("; ".join(problems))
            rel = self.dest_rel(digest)
            dest = ctx.data / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                _remove(dest)
            os.replace(obj, dest)
            if self.digest(dest) != staged:  # the object in place must be the object that was verified
                _remove(dest)
                raise SetupError(f"{self.name}: the object changed while it was moved into place; nothing was placed")
        return {
            "pinned": not problems,
            "pin_id": self.pin_id(),
            "path": rel,
            "mode": mode,
            "origin": origin,
            "digests": digest,
            "installed_utc": _now(),
            "mk_version": _mk_version(),
        }


class FcgGo(Component):
    name = "fcg-go"
    optional = frozenset({"zip_sha256"})

    def expected(self) -> dict[str, Any]:
        return {"zip_sha256": self.pin["zip_sha256"], "binary_sha256": self.pin["binary_sha256"]}

    def digest(self, obj: Path) -> dict[str, Any]:
        return {"binary_sha256": sha256_file(obj)}

    def _binary_from_zip(self, zip_path: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        member, binary = self.pin["member"], stage / "FCG Go"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if member not in zf.namelist():
                    raise SetupError(f"{zip_path.name}: no member {member!r} (has {zf.namelist()[:5]})")
                with zf.open(member) as src, binary.open("wb") as out:
                    shutil.copyfileobj(src, out)
        except zipfile.BadZipFile as exc:
            raise SetupError(f"{zip_path}: not a zip archive ({exc})") from None
        binary.chmod(int(self.pin["mode"], 8))
        return binary, {"zip_sha256": sha256_file(zip_path)}

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        try:
            import pyfcg.fcg_go_bridge as bridge
        except ImportError:
            raise SetupError("pyfcg is not installed: pip install 'memory-kernel[build]'") from None
        target, url = stage / "pyfcg-fcg-go", self.pin["zip_url"]  # must not exist: download_fcg_go deletes it
        try:
            with _pyfcg_context(stage):
                zip_file = bridge.download_fcg_go(str(target), url=url)
        except (urllib.error.URLError, OSError) as exc:
            raise SetupError(f"download of {url} failed: {type(exc).__name__}: {exc}") from None
        if not zip_file or not Path(zip_file).is_file():
            raise SetupError(f"download of {url} failed: pyfcg returned no file")
        binary, facts = self._binary_from_zip(Path(zip_file), stage)
        return binary, facts, url

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        if src.is_dir():
            src = src / self.pin["member"]
        if not src.is_file():
            raise SetupError(f"--adopt-fcg-go {src}: not a file (the binary, its directory, or the zip)")
        if src.suffix == ".zip":
            return self._binary_from_zip(src, stage)
        binary = stage / "FCG Go"
        shutil.copyfile(src, binary)
        binary.chmod(int(self.pin["mode"], 8))
        return binary, {}

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        return f"fcg-go/{digest['binary_sha256'][:12]}/{self.pin['member']}"

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        if not path.is_file() or not os.access(path, os.X_OK):
            return f"{path} is not an executable file: run `mk setup --component {self.name}`"
        return None


class FcgImage(Component):
    name = "fcg-image"

    def expected(self) -> dict[str, Any]:
        return {"sha256": self.pin["sha256"]}

    def digest(self, obj: Path) -> dict[str, Any]:
        return {"sha256": sha256_file(obj)}

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        try:
            import pyfcg.resources as resources
        except ImportError:
            raise SetupError("pyfcg is not installed: pip install 'memory-kernel[build]'") from None
        target = stage / "pyfcg-resources"
        target.mkdir()  # fresh and empty: load_resource skips a file that already exists, unchecked
        try:
            with _pyfcg_context(stage):
                resources.load_resource(self.pin["name"], str(target))
        except (urllib.error.URLError, OSError) as exc:
            raise SetupError(f"download of {self.pin['url']} failed: {type(exc).__name__}: {exc}") from None
        image = target / self.pin["name"]
        if not image.is_file():
            raise SetupError(f"download of {self.pin['url']} failed: pyfcg left no file")
        return image, {}, self.pin["url"]

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        if not src.is_file():
            raise SetupError(f"--adopt-fcg-image {src}: not a file")
        image = stage / self.pin["name"]
        shutil.copyfile(src, image)
        return image, {}

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        return f"fcg-go/resources/{self.pin['name']}"

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        return None if path.is_file() else f"{path} is not a file: run `mk setup --component {self.name}`"


class PropbankFrames(Component):
    name = "propbank-frames"

    def expected(self) -> dict[str, Any]:
        return {
            "xml_manifest_sha256": self.pin["xml_manifest_sha256"],
            "xml_files": self.pin["xml_files"],
            "license_sha256": self.pin["license_sha256"],
        }

    def digest(self, obj: Path) -> dict[str, Any]:
        license_file = obj / "LICENSE"
        manifest, count = manifest_sha256(obj / "frames", suffix=".xml") if (obj / "frames").is_dir() else ("", 0)
        return {
            "xml_manifest_sha256": manifest,
            "xml_files": count,
            "license_sha256": sha256_file(license_file) if license_file.is_file() else "",
        }

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        url = self.pin["archive_url"]
        archive = stage / "archive.tar.gz"
        download(url, archive)  # the archive bytes are not stable: only the extracted manifest is pinned
        obj = stage / "obj"
        (obj / "frames").mkdir(parents=True)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = _safe_parts(member.name, "propbank archive")
                    rest = parts[1:]  # the first part is the archive's top dir
                    if rest == ("LICENSE",):
                        target = obj / "LICENSE"
                    elif len(rest) == 2 and rest[0] == "frames" and rest[1].endswith(".xml"):
                        target = obj / "frames" / rest[1]
                    else:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    with source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
        except tarfile.TarError as exc:
            raise SetupError(f"{url}: not a tar.gz archive ({exc})") from None
        return obj, {}, url

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        root = src.parent if src.name == "frames" and src.is_dir() else src
        if not (root / "frames").is_dir():
            raise SetupError(f"--adopt-frames {src}: no frames/ directory (give the checkout or its frames dir)")
        if not (root / "LICENSE").is_file():
            raise SetupError(
                f"--adopt-frames {src}: no LICENSE next to frames/ (the license text travels with the data)"
            )
        obj = stage / "obj"
        (obj / "frames").mkdir(parents=True)
        for xml in sorted((root / "frames").glob("*.xml")):
            shutil.copyfile(xml, obj / "frames" / xml.name)
        shutil.copyfile(root / "LICENSE", obj / "LICENSE")
        return obj, {}

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        return f"propbank-frames/{self.pin['commit'][:12]}"

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        if not (path / "frames").is_dir() or not (path / "LICENSE").is_file():
            return f"{path} lacks frames/ or LICENSE: run `mk setup --component {self.name}`"
        return None

    def install(self, ctx: Context, adopt: Path | None) -> dict[str, Any]:
        entry = super().install(ctx, adopt)
        (ctx.data / entry["path"] / "COMMIT").write_text(self.pin["commit"] + "\n")
        return entry


class Wordnet(Component):
    name = "wordnet"
    optional = frozenset({"zip_sha256"})

    def expected(self) -> dict[str, Any]:
        return {
            "zip_sha256": self.pin["zip_sha256"],
            "manifest_sha256": self.pin["manifest_sha256"],
            "files": self.pin["files"],
        }

    def digest(self, obj: Path) -> dict[str, Any]:
        manifest, count = manifest_sha256(obj)
        return {"manifest_sha256": manifest, "files": count}

    def _extract(self, zip_path: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        obj = stage / "wordnet"
        obj.mkdir()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    parts = _safe_parts(info.filename, "wordnet zip")
                    if info.is_dir():
                        continue
                    if len(parts) != 2 or parts[0] != "wordnet":
                        raise SetupError(
                            f"{zip_path.name}: unexpected member {info.filename!r} (expected wordnet/<file>)"
                        )
                    with zf.open(info) as src, (obj / parts[1]).open("wb") as out:
                        shutil.copyfileobj(src, out)
        except zipfile.BadZipFile as exc:
            raise SetupError(f"{zip_path}: not a zip archive ({exc})") from None
        return obj, {"zip_sha256": sha256_file(zip_path)}

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        url, zip_path = self.pin["zip_url"], stage / "wordnet.zip"
        download(url, zip_path)
        obj, facts = self._extract(zip_path, stage)
        return obj, facts, url

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        if src.is_file():
            return self._extract(src, stage)
        for candidate in (src, src / "wordnet", src / "corpora" / "wordnet"):
            if (candidate / "index.noun").is_file():
                obj = stage / "wordnet"
                shutil.copytree(candidate, obj)
                return obj, {}
        raise SetupError(f"--adopt-wordnet {src}: neither wordnet.zip nor a wordnet corpus directory")

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        return "nltk_data/corpora/wordnet"

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        return (
            None
            if (path / "index.noun").is_file()
            else f"{path} is not a wordnet corpus: run `mk setup --component {self.name}`"
        )


class Tiktoken(Component):
    name = "tiktoken"

    def expected(self) -> dict[str, Any]:
        return {"sha256": self.pin["sha256"]}

    def digest(self, obj: Path) -> dict[str, Any]:
        return {"sha256": sha256_file(obj)}

    def cache_key(self) -> str:
        """tiktoken's cache file name: the sha1 of the source URL (``TIKTOKEN_CACHE_DIR`` = ``<data>/tiktoken``)."""
        return hashlib.sha1(self.pin["url"].encode()).hexdigest()

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        target = stage / self.cache_key()
        download(self.pin["url"], target)
        return target, {}, self.pin["url"]

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        if not src.is_file():
            raise SetupError(f"--adopt-tiktoken {src}: not a file")
        target = stage / self.cache_key()
        shutil.copyfile(src, target)
        return target, {}

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        return f"tiktoken/{self.cache_key()}"

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        return None if path.is_file() else f"{path} is not a file: run `mk setup --component {self.name}`"


class ParserModel(Component):
    """The ``benepar_en3`` constituency model the managed parser loads (``memory_kernel.parser_service``, the
    ``[parser]`` extra). Fetched as the pinned zip and unpacked to ``<data>/nltk_data/models/<name>``: the parser runs with
    ``NLTK_DATA=<data>/nltk_data``, where benepar's ``models/<name>`` lookup finds it. Pinned by the zip sha and by the
    sha of each unpacked file (a directory can be adopted, so the files are what is verified)."""

    name = "parser-model"
    optional = frozenset({"zip_sha256"})

    def expected(self) -> dict[str, Any]:
        return {
            "zip_sha256": self.pin["zip_sha256"],
            "manifest_sha256": self.pin["manifest_sha256"],
            "files": dict(self.pin["files"]),
        }

    def digest(self, obj: Path) -> dict[str, Any]:
        manifest, _count = manifest_sha256(obj)
        files = {p.relative_to(obj).as_posix(): sha256_file(p) for p in sorted(obj.rglob("*")) if p.is_file()}
        return {"manifest_sha256": manifest, "files": files}

    def _extract(self, zip_path: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        name = self.pin["name"]
        obj = stage / name
        obj.mkdir()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    parts = _safe_parts(info.filename, "parser model zip")
                    if info.is_dir():
                        continue
                    if len(parts) != 2 or parts[0] != name:
                        raise SetupError(
                            f"{zip_path.name}: unexpected member {info.filename!r} (expected {name}/<file>)"
                        )
                    with zf.open(info) as src, (obj / parts[1]).open("wb") as out:
                        shutil.copyfileobj(src, out)
        except zipfile.BadZipFile as exc:
            raise SetupError(f"{zip_path}: not a zip archive ({exc})") from None
        return obj, {"zip_sha256": sha256_file(zip_path)}

    def fetch(self, stage: Path) -> tuple[Path, dict[str, Any], str]:
        url, zip_path = self.pin["zip_url"], stage / f"{self.pin['name']}.zip"
        download(url, zip_path)
        obj, facts = self._extract(zip_path, stage)
        return obj, facts, url

    def adopt(self, src: Path, stage: Path) -> tuple[Path, dict[str, Any]]:
        if src.is_file():
            return self._extract(src, stage)
        name = self.pin["name"]
        for candidate in (src, src / name, src / "models" / name):
            if (candidate / "benepar_model.bin").is_file():
                obj = stage / name
                shutil.copytree(candidate, obj, symlinks=False)
                return obj, {}
        raise SetupError(f"--adopt-parser-model {src}: neither {name}.zip nor an unpacked {name} directory")

    def dest_rel(self, digest: Mapping[str, Any]) -> str:
        return f"nltk_data/models/{self.pin['name']}"

    def shape(self, path: Path, entry: Mapping[str, Any]) -> str | None:
        return (
            None
            if (path / "benepar_model.bin").is_file()
            else f"{path} holds no parser model: run `mk setup --component {self.name}`"
        )


class SpacyModel(Component):
    """``en_core_web_sm`` lives in the RUNNING interpreter, not in ``<data>``: its state is read from that interpreter's
    installed metadata, and receipts are informational only."""

    name = "spacy-model"

    def expected(self) -> dict[str, Any]:
        return {"manifest_sha256": self.pin["installed_manifest_sha256"], "files": self.pin["files"]}

    def digest(self, obj: Path) -> dict[str, Any]:
        manifest, count = manifest_sha256(obj)
        return {"manifest_sha256": manifest, "files": count}

    @property
    def _package(self) -> str:
        return self.pin["package"]

    def _package_dir(self) -> Path | None:
        spec = importlib.util.find_spec(self._package)
        locations = list(spec.submodule_search_locations or []) if spec else []
        return Path(locations[0]) if locations else None

    def check(self, ctx: Context, entry: Mapping[str, Any] | None) -> Row:
        row = Row(self.name, pinned=None)
        model, spacy = installed_version(self._package), installed_version("spacy")
        if model is None:
            row.detail = f"{self._package} is not installed in {sys.executable}: run `mk setup --component {self.name}`"
            return row
        package_dir = self._package_dir()
        row.path = str(package_dir) if package_dir else None
        if model != self.pin["version"]:
            row.status = MISMATCH
            row.detail = f"{self._package} {model} is installed, {self.pin['version']} is pinned"
            return row
        if spacy != self.pin["spacy_version"]:
            row.status = MISMATCH
            row.detail = f"spacy {spacy} is installed, the model is pinned to spacy {self.pin['spacy_version']}"
            return row
        row.pinned = True
        if ctx.verify:
            if package_dir is None:
                row.detail = f"{self._package} has no package directory"
                return row
            problems = self.compare(self.digest(package_dir))
            if problems:
                row.status, row.detail = MISMATCH, "; ".join(problems)
                return row
        row.status = OK
        row.detail = f"{self._package} {model} with spacy {spacy}" + (", verified" if ctx.verify else "")
        return row

    def pip_command(self) -> list[str]:
        """``--no-deps``: the model's only requirement is spaCy, a base dependency of this package."""
        spec = f"{self.pin['wheel_url']}#sha256={self.pin['wheel_sha256']}"
        return [sys.executable, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", spec]

    def install(self, ctx: Context, adopt: Path | None) -> dict[str, Any]:
        if adopt is not None:
            return self._adopt_into_site_packages(ctx, adopt)
        if importlib.util.find_spec("pip") is None:
            wheel = self.pin["wheel_url"].rsplit("/", 1)[1]
            raise SetupError(
                f"pip is not available in {sys.executable}; install the pinned wheel yourself, then run "
                f"`mk setup --check --verify`:\n"
                f"  curl -fLO {self.pin['wheel_url']}\n"
                f'  echo "{self.pin["wheel_sha256"]}  {wheel}" | sha256sum -c -\n'
                f'  uv pip install --python "{sys.executable}" --no-deps ./{wheel}'
            )
        done = subprocess.run(self.pip_command(), capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)
        if done.returncode != 0:
            tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-6:])
            raise SetupError(f"pip install of the pinned {self._package} wheel failed (rc {done.returncode}):\n{tail}")
        importlib.invalidate_caches()  # the package appeared in this process's site-packages after the finders listed it
        package_dir = self._package_dir()
        digest = self.digest(package_dir) if package_dir else {}
        problems = self.compare(digest)
        if problems:
            raise SetupError("installed, but the files differ from the pin: " + "; ".join(problems))
        return self._entry(package_dir, digest, "download", self.pin["wheel_url"])

    def _entry(self, package_dir: Path | None, digest: Mapping[str, Any], mode: str, origin: str) -> dict[str, Any]:
        return {
            "pinned": True,
            "pin_id": self.pin_id(),
            "path": None,
            "installed_into": str(package_dir) if package_dir else None,
            "interpreter": sys.executable,
            "mode": mode,
            "origin": origin,
            "digests": dict(digest),
            "installed_utc": _now(),
            "mk_version": _mk_version(),
        }

    def _adopt_into_site_packages(self, ctx: Context, src: Path) -> dict[str, Any]:
        if not (src / "meta.json").is_file():
            raise SetupError(f"--adopt-spacy-model {src}: not an installed {self._package} package (no meta.json)")
        dist_infos = sorted(src.parent.glob(f"{self._package}-{self.pin['version']}.dist-info"))
        if not dist_infos:
            raise SetupError(
                f"--adopt-spacy-model {src}: no {self._package}-{self.pin['version']}.dist-info next to it "
                "(the installed metadata is how `mk setup --check` finds the model)"
            )
        purelib = Path(sysconfig.get_paths()["purelib"])
        dest = purelib / self._package
        if installed_version(self._package) is not None or dest.exists():
            current = self.check(Context(ctx.data, verify=True), None)
            if current.status == OK:
                return self._entry(dest, self.digest(dest), "adopt", str(src))
            raise SetupError(
                f"{self._package} is already installed in {sys.executable} ({current.detail}); uninstall it first"
            )
        with staging(ctx.data, self.name) as stage:
            copy = stage / self._package
            shutil.copytree(src, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            digest = self.digest(copy)
            problems = self.compare(digest)
            if problems:
                raise SetupError("; ".join(problems))
            info_copy = stage / dist_infos[0].name
            shutil.copytree(dist_infos[0], info_copy)
            (info_copy / "INSTALLER").write_text("mk-setup\n")
            # the final move is inside purelib (same filesystem as the destination): verified bytes, atomic rename
            hold = Path(tempfile.mkdtemp(prefix=".mk-setup-", dir=purelib))
            try:
                shutil.move(str(copy), hold / self._package)
                shutil.move(str(info_copy), hold / info_copy.name)
                os.replace(hold / info_copy.name, purelib / info_copy.name)
                os.replace(hold / self._package, dest)
            finally:
                shutil.rmtree(hold, ignore_errors=True)
        importlib.invalidate_caches()
        return self._entry(dest, digest, "adopt", str(src))


class SpacyModelTrf(SpacyModel):
    """``en_core_web_trf``: the transformer pipeline the runtime parses every turn with by default. Installed like the
    small model (``--no-deps``), with no ``--adopt-*`` option. It loads only where
    ``memory-kernel[parser]`` is installed (``spacy-curated-transformers``, torch, transformers), so ``check`` also
    loads it: this component is never OK when it cannot load. Without the extra it does not apply at all: ``mk setup``
    skips it and ``--check`` reports OK with a note (serve and query fall back to the small model; builds need the
    extra)."""

    name = "spacy-model-trf"

    def skip_note(self) -> str | None:
        if parser_extra_installed():
            return None
        return (
            "skipped: trf loads only with `pip install 'memory-kernel[parser]'` (serve and query fall back to "
            "en_core_web_sm; builds need the extra); opt out of trf with MK_EXTRACTION_SPACY_MODEL=en_core_web_sm"
        )

    def check(self, ctx: Context, entry: Mapping[str, Any] | None) -> Row:
        note = self.skip_note()
        if note is not None:
            return Row(self.name, status=OK, detail=note)
        row = super().check(ctx, entry)
        if row.status != OK:
            return row
        try:
            import spacy

            spacy.load(self._package)
        except Exception as exc:  # any failure to load is a failure of the component
            row.status = MISMATCH
            row.detail = (
                f"{self._package} is installed but does not load ({type(exc).__name__}: {exc}): "
                "install the parser extra, `pip install 'memory-kernel[parser]'`"
            )
        return row


_CLASSES: dict[str, type[Component]] = {
    "fcg-go": FcgGo,
    "fcg-image": FcgImage,
    "propbank-frames": PropbankFrames,
    "spacy-model": SpacyModel,
    "spacy-model-trf": SpacyModelTrf,
    "wordnet": Wordnet,
    "tiktoken": Tiktoken,
    "parser-model": ParserModel,
}


def components(pins: Mapping[str, Any] | None = None) -> dict[str, Component]:
    pins = load_pins() if pins is None else pins
    return {name: cls(pins["components"][name]) for name, cls in _CLASSES.items()}


# -- the command ------------------------------------------------------------------------------------------------------------


@dataclass
class Request:
    data: Path
    components: Sequence[str] = ()
    check: bool = False
    verify: bool = False
    offline: bool = False
    allow_unpinned_engine: bool = False
    adopt: Mapping[str, Path] = field(default_factory=dict)


def check_platform(pins: Mapping[str, Any]) -> None:
    want = pins["platform"]
    system, machine = platform.system(), platform.machine()
    if system != want["system"] or machine not in (want["machine"], "AMD64"):
        raise SetupError(
            f"mk setup supports {want['system']} {want['machine']} only (the pins exist for that platform); "
            f"this is {system} {machine}"
        )


def _validate(req: Request) -> None:
    if req.verify and not req.check:
        raise SetupError("--verify belongs to --check")
    if req.check and req.adopt:
        raise SetupError("--check never changes anything: it cannot be combined with --adopt-*")
    if req.adopt and not req.offline:
        raise SetupError("--adopt-* is for --offline use: add --offline (an online run downloads instead)")
    for name in req.components:
        if name not in COMPONENTS:
            raise SetupError(f"unknown component {name!r} (choose from {', '.join(COMPONENTS)})")


def run(req: Request, *, as_json: bool, out: Any, err: Any) -> int:
    """Run ``mk setup``: 0 = every selected component verifies, 1 = something is missing or differs, 2 = a refusal."""
    _validate(req)
    pins = load_pins()
    check_platform(pins)
    comps = components(pins)
    selected = list(req.components) or [n for n in COMPONENTS if n in req.adopt] or list(COMPONENTS)
    stray = sorted(set(req.adopt) - set(selected))
    if stray:
        raise SetupError(f"--adopt-* for {', '.join(stray)}, which --component leaves out")
    receipt = read_receipt(req.data)
    ctx = Context(req.data, allow_unpinned=req.allow_unpinned_engine, verify=req.verify)

    rows: list[Row] = []
    for name in selected:
        comp = comps[name]
        entry = receipt["components"].get(name)
        if comp.skip_note() is not None and name not in req.adopt:  # does not apply here: OK, with its note
            rows.append(comp.check(ctx, entry))
            continue
        if req.check or (req.offline and name not in req.adopt):
            rows.append(comp.check(ctx, entry))
            continue
        try:
            if name in req.adopt:
                action, new = "adopted", comp.install(ctx, req.adopt[name])
            else:
                current = comp.check(Context(req.data, verify=True), entry)
                kept_unpinned = (entry or {}).get("pinned") is False and req.allow_unpinned_engine
                if current.status == OK and (current.pinned or kept_unpinned):
                    current.action = "unchanged"
                    rows.append(current)
                    continue
                action, new = "downloaded", comp.install(ctx, None)
            receipt["components"][name] = new
            _write_receipt(req.data, receipt)
        except (SetupError, OSError) as exc:  # one component failing does not stop the others
            row = comp.check(Context(req.data), entry)
            row.error = str(exc) if isinstance(exc, SetupError) else f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row = comp.check(Context(req.data), new)  # install() verified the staged object and the one in place
        row.action = action
        if row.status == OK:
            row.detail = f"{action} and verified"
        rows.append(row)

    ok = all(r.status == OK and r.error is None for r in rows)
    code = 2 if any(r.error for r in rows) else (0 if ok else 1)
    _report(rows, req, as_json=as_json, ok=ok, out=out, err=err)
    return code


def _report(rows: Sequence[Row], req: Request, *, as_json: bool, ok: bool, out: Any, err: Any) -> None:
    for row in rows:
        if row.error:
            print(f"{row.name}: refused: {row.error}", file=err)
        elif row.status != OK:
            print(f"{row.name}: {row.status}: {row.detail}", file=err)
    if as_json:
        doc = {
            "command": "check" if req.check else "setup",
            "data_dir": str(req.data),
            "ok": ok,
            "components": [r.to_json() for r in rows],
        }
        print(json.dumps(doc, indent=1, sort_keys=True), file=out)
        return
    print(f"data dir: {req.data}", file=out)
    print(f"{'component':<16} {'status':<9} {'pinned':<7} location / detail", file=out)
    for row in rows:
        pinned = "-" if row.pinned is None else ("yes" if row.pinned else "NO")
        shown = row.action + ": " if row.action and row.action != "unchanged" else ""
        print(f"{row.name:<16} {row.status:<9} {pinned:<7} {shown}{row.path or row.detail}", file=out)
