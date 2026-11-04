"""Hazard ratchet: no private identifier may appear in any tracked file, THIS FILE INCLUDED.

``BANNED`` holds only SALTED sha256 digests of normalised tokens (never the tokens), so the list itself publishes
nothing. Every tracked file, and every tracked PATH, is tokenised; each token is hashed and looked up. A hit fails
the test with ``path:line`` and a digest prefix, never the offending text. Generic shapes (absolute home paths,
IPv4 addresses, long numeric ids, e-mail addresses, API keys, PEM headers) are caught by structural rules instead
of digests, because a digest of a short numeric id would only be a brute-force target.

Normalisation (see ``tokens``): NFKD, accents and zero-width characters stripped, lower-cased; runs of [A-Za-z0-9]
are the unit, and a name is the same token however it is spelled: ``foo-bar``, ``foo_bar``, ``FooBar`` and
``foo.bar`` all give ``foobar``; adjacent pieces of a longer camel-case run join the same way (``FooBarBaz`` also gives
``foobar`` and ``barbaz``). Letter/digit boundaries split (``name10`` gives ``name`` and ``10``). Names joined
by ``::`` or ``:x`` (Lisp package qualifiers) carry a ``:`` suffix and are therefore not the bare name.

Exemptions are digests too, and each one is a visible, reviewable line:

* ``ALLOWED_CHAINS``: digests of whole id/path chains (the canonical concatenated form) that must stay
  byte-identical because sealed replay inputs carry them;
* ``ALLOWED_LINES``: digests of whole normalised lines (sealed free-text literals);
* ``SEALED_FILES``: path digest to content sha256 for files whose bytes are sealed. Their tokens are exempt, but
  ANY change of the content fails the test ("sealed file changed: re-scrub or re-pin").

The ratchet covers TRACKED files only (the export set). The private sweep that decodes a hit, folds adjacent
literals, searches plain substrings and scans git metadata is not part of this repository.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SALT = bytes.fromhex("48dde18e17f2b70e1fde9be36eae48e9")
DIGEST_HEX = 32
#: digest -> scope.  "ALL": every tracked file.  "NOPROSE": every file except the prose files in PROSE.
BANNED: dict[str, str] = {
    "03037d1c1e67d3afd57945c909bb4007": "ALL",
    "05e7ed3338bf4409335407b82bdd0922": "ALL",
    "0712fdbfab70d2229a4acfa35a04a929": "ALL",
    "0b1d12f65567fa7f57c68777bcc9b1eb": "ALL",
    "10ebda3af5d0694e43f2ee7cc8d691c6": "ALL",
    "32a3302320ae795d81a258cb758070c6": "ALL",
    "32cd82bd739fa91eb7a47d5a64b7ee8b": "ALL",
    "389a9259f6daef4d1232d04612c5c147": "ALL",
    "4d3626deea4882f9806894c516791e1a": "ALL",
    "51a680583a2e2b5e022091693cf6a5eb": "ALL",
    "610bc7762120ecf55bfa4990ff4724fb": "ALL",
    "6ce765a7c21b943b7c83db7f128bd4e6": "ALL",
    "80546c9c9070550d40ab775865da554e": "ALL",
    "8ccab79a140a0e8def4b3672ec78cf6a": "ALL",
    "939a1c6ada33d3580d3c5762f68d1de3": "ALL",
    "97e3a7fbf19301d5cfcd13ac3b9afca9": "ALL",
    "a0ac66ce0e9b53ebdc8d2018fe516913": "ALL",
    "a162839d2fe778c80c710b5b8cee2681": "ALL",
    "af8ca19caad9e0999578d6bc683777fa": "ALL",
    "c48d236e6bad910619a785a2c449fedd": "ALL",
    "c9710273d73e68882d4754815023caef": "ALL",
    "cb3088a14984c5f5cb2bb0ea07acdbdd": "ALL",
    "e54e09b03dec3227d775249b2a4db35c": "ALL",
    "e93c610adc4c0b24b2eb5f6c2c57c021": "ALL",
    "ecdc7daabb21b4a4c006042e92db8a60": "NOPROSE",
    "edba23c575583c60f81058d883da86f6": "ALL",
    "fa57d90364f260f76ce2200b110e19d8": "ALL",
}
#: digests of whole chains (``chain:`` namespace) whose tokens are skipped.
ALLOWED_CHAINS: frozenset[str] = frozenset()
#: digests of whole normalised lines (``line:`` namespace) whose tokens are skipped.
ALLOWED_LINES: frozenset[str] = frozenset(
    {
        # data/runtime-grammar/question-map.json: one upstream PropBank roleset id (a public English verb) shares its
        # lemma with a banned token; the generated line is exempt as a whole, so any regeneration changing it fails here
        # (re-pinned in Task 11a: the line gained its arg1's store kind, Place)
        "29e6389726f75a70baf6add4310961cc",
    }
)
#: path digest (``path:`` namespace) -> sha256 of the file's bytes.
SEALED_FILES: dict[str, str] = {}
PROSE = ("README.md", "CHANGELOG.md", "NOTICE", "CONTRIBUTING.md", "docs/architecture.md")

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_CHAIN = re.compile(r"[A-Za-z0-9]+(?:[-_./\\][A-Za-z0-9]+)*")
_RUN = re.compile(r"[A-Za-z0-9]+")
_SUB = re.compile(r"[a-z]+|[0-9]+")
_CAMEL = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_HEX = re.compile(r"[0-9a-f]+")
MAX_N, MAX_RUN = 4, 64

# Generic, non-private shapes.  The [x] trick keeps each pattern from matching its own source text.
STRUCTURAL = {
    "abs-home": re.compile(r"(?<![\w.])/[h]ome/[\w.-]+"),
    "mac-home": re.compile(r"(?<![\w.])/[U]sers/[\w.-]+"),
    "ipv4": re.compile(r"(?<!\w)(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?!\w)(?!\.\d)"),
    # a digit run preceded by "." is the tail of a decimal fraction (package data), not an id
    "long-id": re.compile(r"(?<![\w.])\d{9,}(?!\w)(?!\.\d)"),
    "object-id": re.compile(r"(?<![\w.])[0-9a-f]{24}(?![\w.])"),
    "email": re.compile(r"[\w.+-]+@(?!example\.)[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"),
    "api-key": re.compile(
        r"\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}"
    ),
    "pem": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
}
IPV4_OK = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}
LOCKFILE_SKIPS: dict[str, set[str]] = {"uv.lock": {"ipv4", "long-id", "object-id"}}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).translate(_ZW)
    return "".join(c for c in text if not unicodedata.combining(c))


def chains(line: str) -> Iterator[tuple[str, set[str]]]:
    """Yield ``(canonical chain, its tokens)`` for every chain of one line.

    The canonical chain is the lower-cased runs concatenated without separators; it is what ``ALLOWED_CHAINS`` hashes.
    """
    line = _fold(line)
    for m in _CHAIN.finditer(line):
        chain = m.group(0)
        nxt = line[m.end() : m.end() + 2]
        qualified = nxt.startswith("::") or (nxt[:1] == ":" and nxt[1:2].isalpha())
        out: set[str] = set()
        parts: list[str] = []
        ok = True
        runs = list(_RUN.finditer(chain))
        for rm in runs:
            raw = rm.group(0)
            if len(raw) > MAX_RUN:
                ok = False
                continue
            low = raw.lower()
            out.add(low + (":" if qualified and rm is runs[-1] else ""))
            if len(low) >= 16 and _HEX.fullmatch(low):  # hash / object-id blob: opaque
                parts.append(low)
                continue
            pieces = _CAMEL.findall(raw)
            if len(pieces) > 1:  # a real camel split; a single piece is the run itself (qualifier-aware above)
                lowered = [piece.lower() for piece in pieces]
                out.update(lowered)
                for n in range(2, MAX_N + 1):  # adjacent pieces joined, as for the parts of a separated chain
                    out.update("".join(lowered[i : i + n]) for i in range(len(lowered) - n + 1))
            subs = _SUB.findall(low)
            if len(subs) > 1:
                out.update(subs)
            parts.extend(subs)
        if ok:
            for n in range(2, MAX_N + 1):
                for i in range(len(parts) - n + 1):
                    out.add("".join(parts[i : i + n]) + (":" if qualified and i + n == len(parts) else ""))
        yield "".join(rm.group(0).lower() for rm in runs), out


def tokens(line: str) -> set[str]:
    out: set[str] = set()
    for _, toks in chains(line):
        out |= toks
    return out


def digest(token: str) -> str:
    return hashlib.sha256(SALT + token.encode()).hexdigest()[:DIGEST_HEX]


def chain_digest(canon: str) -> str:
    return digest("chain:" + canon)


def line_digest(line: str) -> str:
    return digest("line:" + " ".join(_fold(line).lower().split()))


def path_digest(rel: str) -> str:
    return digest("path:" + rel)


def scan_text(
    path: str,
    text: str,
    banned: dict[str, str] | None = None,
    allowed_chains: frozenset[str] | None = None,
    allowed_lines: frozenset[str] | None = None,
) -> list[tuple[int, str]]:
    """[(line, rule)] for one file's text.  rule is ``tok:<digest prefix>`` or ``re:<structural name>``."""
    banned = BANNED if banned is None else banned
    allowed_chains = ALLOWED_CHAINS if allowed_chains is None else allowed_chains
    allowed_lines = ALLOWED_LINES if allowed_lines is None else allowed_lines
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        if line_digest(line) not in allowed_lines:
            for canon, toks in chains(line):
                if chain_digest(canon) in allowed_chains:
                    continue
                for tok in toks:
                    d = digest(tok)
                    scope = banned.get(d)
                    if scope == "ALL" or (scope == "NOPROSE" and path not in PROSE):
                        hits.append((lineno, "tok:" + d[:8]))
        for name, rx in STRUCTURAL.items():
            if name in LOCKFILE_SKIPS.get(path, ()):
                continue
            for m in rx.finditer(line):
                if name == "ipv4" and m.group(0) in IPV4_OK:
                    continue
                hits.append((lineno, "re:" + name))
    return hits


def tracked_files(root: Path = ROOT) -> list[str]:
    if (root / ".git").exists():
        raw = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
        return [p for p in raw.decode().split("\0") if p]
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "dist", "build"}
    found = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip and not d.endswith(".egg-info")]
        found += [(Path(base) / n).relative_to(root).as_posix() for n in names]
    return sorted(found)


def read_text(p: Path) -> tuple[bytes, str]:
    raw = p.read_bytes()
    return raw, raw.decode("latin-1") if b"\0" in raw[:8192] else raw.decode("utf-8", errors="replace")


def scan_tree(root: Path = ROOT, banned: dict[str, str] | None = None) -> list[str]:
    problems: list[str] = []
    for rel in tracked_files(root):
        p = root / rel
        if p.is_symlink() or not p.is_file():
            continue
        raw, text = read_text(p)
        sealed = SEALED_FILES.get(path_digest(rel))
        if sealed is not None:
            if hashlib.sha256(raw).hexdigest() != sealed:
                problems.append(f"sealed:{path_digest(rel)[:8]}:0 sealed file changed: re-scrub or re-pin")
            # tokens are exempt for a sealed file whose bytes are pinned; the structural rules still run
            problems += [f"sealed:{path_digest(rel)[:8]}:{n} {rule}" for n, rule in scan_text(rel, text, {})]
            continue
        problems += [f"{rel}:0 {rule}" for _, rule in scan_text(rel, "\n" + rel, banned)]  # the PATH itself
        problems += [f"{rel}:{lineno} {rule}" for lineno, rule in scan_text(rel, text, banned)]
    return problems


def test_no_hazard_token_in_any_tracked_file() -> None:
    problems = scan_tree()
    assert not problems, "hazard ratchet hits (private tokens/shapes; see the private sweep to decode):\n" + "\n".join(
        problems[:200]
    )


def test_this_file_is_scanned_with_no_exemption() -> None:
    rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert rel in tracked_files(), "the ratchet must be a tracked file so that it scans itself"
    assert path_digest(rel) not in SEALED_FILES
    source = Path(__file__).read_text(encoding="utf-8")
    assert scan_text(rel, source) == []
    # and again with every exemption list emptied: no exemption is doing any work for this file
    assert scan_text(rel, source, allowed_chains=frozenset(), allowed_lines=frozenset()) == []
    assert scan_text(rel, "\n" + rel, allowed_chains=frozenset(), allowed_lines=frozenset()) == []


def test_canary_positive_control() -> None:
    parts = ("zq9", "canary", "tok")
    canary = "".join(parts)  # built at runtime: the contiguous word never appears in this file (asserted below)
    hyphen = "-".join(parts[:2]).title() + "_" + parts[2].title()  # title-cased, joined by a hyphen and an underscore
    zero_width = (parts[0] + parts[1]).upper() + "\u200b" + parts[2].upper()
    banned = {digest(canary): "ALL"}
    spellings = (canary, hyphen, "".join(p.title() for p in parts), ".".join(parts), zero_width, "x " + canary + "'s")
    for spelling in spellings:
        assert scan_text("x.txt", spelling, banned, frozenset(), frozenset()), spelling
    assert not scan_text("x.txt", "zq9 canary tokens", banned)  # spaces split names; plural is another word
    assert not scan_text("x.txt", "ns::" + canary + "::y", banned)  # a Lisp package qualifier is not the bare name
    # the exemption plumbing: a whole-chain digest, a whole-line digest
    chain_ok = frozenset({chain_digest(canary)})
    assert not scan_text("x.txt", hyphen, banned, chain_ok, frozenset())
    assert scan_text("x.txt", hyphen + "-extra", banned, chain_ok, frozenset())  # a longer chain is another chain
    line = "say  " + canary + " twice"
    assert not scan_text("x.txt", line, banned, frozenset(), frozenset({line_digest("say " + canary + " twice")}))
    # the comment above is true: this very file, scanned with the canary banned and nothing exempt, holds no hit
    assert scan_text("tests/test_hazard_ratchet.py", Path(__file__).read_text(encoding="utf-8"), banned) == []


def test_camel_case_pieces_join_like_the_parts_of_a_separated_chain() -> None:
    parts = ("zq9", "canary", "tok")
    banned = {digest("".join(parts[:2])): "ALL"}
    camel = "".join(p.title() for p in parts)  # the banned name is the first two pieces of a longer camel-case run
    assert scan_text("x.txt", camel, banned)
    assert scan_text("x.txt", "x" + camel + "Suffix", banned)
    assert scan_text("x.txt", "_".join(parts), banned)
    assert not scan_text("x.txt", parts[0].title() + parts[2].title() + parts[1].title(), banned)  # not adjacent


def test_retired_subsystem_name_and_phase_tag_stay_banned() -> None:
    """Two digests the private sweep emits (``--emit-hashes``, kind ``retired-name``): the retired subsystem name and the
    retired phase tag. No test spells either, so the digests are pinned here rather than rebuilt."""
    retired = ("4d3626deea4882f9806894c516791e1a", "c48d236e6bad910619a785a2c449fedd")
    assert [BANNED.get(d) for d in retired] == ["ALL", "ALL"]


def test_sealed_file_pin(tmp_path: Path) -> None:
    """A sealed file's tokens are exempt only while its bytes match the pin."""
    canary = "zq9" + "canary" + "tok"
    (tmp_path / canary).mkdir()
    rel = canary + "/data.txt"
    f = tmp_path / rel
    f.write_text("alpha " + canary + "\n")
    banned = {digest(canary): "ALL"}
    assert scan_tree(tmp_path, banned)  # unsealed: the path and the content both fire
    pinned = {path_digest(rel): hashlib.sha256(f.read_bytes()).hexdigest()}
    original = dict(SEALED_FILES)
    try:
        SEALED_FILES.update(pinned)
        assert scan_tree(tmp_path, banned) == []
        f.write_text("alpha " + canary + " changed\n")
        assert any("sealed file changed" in p for p in scan_tree(tmp_path, banned))
    finally:
        SEALED_FILES.clear()
        SEALED_FILES.update(original)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("abs-home", "/" + "home/someone/project"),
        ("mac-home", "/" + "Users/someone/project"),
        ("ipv4", "10." + "1.2.3"),
        ("long-id", "1" + "2" * 9),
        ("long-id", "see " + "7" * 10 + "."),  # a sentence-final full stop must not hide an id
        ("ipv4", "host 10." + "1.2.3."),
        ("email", "a.b" + "@" + "corp.example.org"),
        ("api-key", "sk" + "-" + "a" * 24),
    ],
)
def test_structural_rules_fire(name: str, text: str) -> None:
    assert any(rule == "re:" + name for _, rule in scan_text("x.txt", text, {}))


def test_long_id_skips_decimal_fraction_tails() -> None:
    assert not any(rule == "re:long-id" for _, rule in scan_text("x.json", "0." + "1" * 9, {}))
    assert not any(rule == "re:long-id" for _, rule in scan_text("x.json", "12." + "3" * 10, {}))
    assert any(rule == "re:long-id" for _, rule in scan_text("x.json", "id " + "1" * 9, {}))
