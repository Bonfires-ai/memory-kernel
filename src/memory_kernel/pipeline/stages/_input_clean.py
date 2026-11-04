"""Input cleanup and splitting: the parser's limits (2,000 characters per part, 31 trees, prose only) are met before
the text reaches it. A text with nothing to clean comes back unchanged."""

import html
import re
from urllib.parse import urlsplit

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_DATA_URI = re.compile(r"(?<![\w])data:[\w/+.\-]+(?:;[\w=.\-]+)*,\S*", re.IGNORECASE)
_HTML_TAG = re.compile(
    r"</?(?:html|head|body|div|p|span|a|br|table|tr|td|ul|ol|li|script|style|b|i|h[1-6])(?:\s*/?>|\s+[\w:-]+\s*=)", re.I
)
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_BLOB = re.compile(r"\S{80,}")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _host(url: str) -> str:
    full = url if "://" in url else "http://" + url
    try:
        host = urlsplit(full).hostname or "link"
    except ValueError:  # e.g. an unbalanced bracket in the authority
        return "link"
    return host.removeprefix("www.")


def _link(match: re.Match[str]) -> str:
    url = match.group(0)
    body = url.rstrip(".,;:!?")  # sentence punctuation after the link is not part of it
    return f"[link: {_host(body)}]{url[len(body) :]}"


def clean_text(text: str) -> str:
    out = _DATA_URI.sub("[embedded web page]", text)
    if _HTML_TAG.search(out):
        out = html.unescape(_TAG.sub(" ", _SCRIPT.sub(" ", out)))
    out = _URL.sub(_link, out)
    out = _BLOB.sub("[blob]", out)
    if out == text:
        return text
    return re.sub(r"\s+", " ", out).strip()


def split_parts(text: str, *, max_chars: int = 1900, max_sentences: int = 20) -> list[str]:
    sentences = _SENTENCE.split(text)
    if len(text) <= max_chars and len(sentences) <= max_sentences:
        return [text]
    pieces: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        pieces.append(sentence)
    parts: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = " ".join([*current, piece])
        if current and (len(candidate) > max_chars or len(current) >= max_sentences):
            parts.append(" ".join(current))
            current = [piece]
        else:
            current.append(piece)
    if current:
        parts.append(" ".join(current))
    return parts
