"""WordNet-backed semantic composition for normalized statements."""

from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from memory_kernel.grammar.dependencies import WORDNET_LOOKUP_LOCK as _WORDNET_LOCK
from memory_kernel.grammar.dependencies import load_spacy_model
from memory_kernel.grammar.dependencies import wordnet_module as _shared_wordnet_module
from memory_kernel.primitives import english_stopwords as _english_stopwords

logger = logging.getLogger(__name__)


class WordNetSemantics:
    """Structural noun semantics backed by WordNet and lemmatized ngrams."""

    def candidate_terms(self, text: str) -> tuple[str, ...]:
        return _candidate_wordnet_terms(text)

    def compound_role_action_terms(self, text: str, *, max_terms: int = 12) -> tuple[str, ...]:
        return _wordnet_compound_role_action_terms(text, max_terms=max_terms)

    def taxonomy_metadata(self, text: str, *, max_terms: int = 12) -> dict[str, list[str]]:
        return _wordnet_taxonomy_metadata_for_text(text, max_terms=max_terms)

    def term_lexnames(self, term: str) -> frozenset[str]:
        return _wordnet_term_lexnames(term)

    def noun_synset_shapes(self, term: str) -> tuple[tuple[str, int], ...]:
        return _wordnet_noun_synset_shapes(term)

    def noun_sense_lexnames(self, term: str) -> tuple[str, ...]:
        return _wordnet_noun_sense_lexnames(term)

    def first_noun_sense_closure(self, term: str) -> frozenset[str]:
        return _wordnet_first_noun_sense_closure(term)

    def noun_senses_under(self, term: str, concept: str) -> bool | None:
        return _wordnet_noun_senses_under(term, concept)

    def verb_lexnames(self, term: str) -> frozenset[str]:
        return _wordnet_verb_lexnames(term)

    def primary_verb_lexname(self, term: str) -> str:
        return _wordnet_primary_verb_lexname(term)

    def verb_lemma(self, term: str) -> str:
        return _wordnet_verb_lemma(term)

    def derivationally_related_verbs(self, term: str) -> tuple[str, ...]:
        return _wordnet_derivationally_related_verbs(term)

    def adjective_nominal_terms(self, term: str) -> tuple[str, ...]:
        return _wordnet_adjective_nominal_terms(term)

    def duration_unit_term(self, term: str) -> bool:
        return _wordnet_duration_unit_term(term)

    def lookup_variants(self, term: str) -> tuple[str, ...]:
        return _wordnet_lookup_variants(term)


@lru_cache(maxsize=1)
def get_wordnet_semantics() -> WordNetSemantics:
    return WordNetSemantics()


def _wordnet() -> Any | None:
    """Return the wordnet corpus module if ready, else ``None``.

    Delegates to `grammar.dependencies.wordnet_module()`, the single shared
    readiness-probe seam (error-once, cached for the process), instead of this
    module keeping its own independent load guard. `wordnet_ready()`'s probe forces
    the corpus load via a trivial `synsets("test")` call and yields ``None`` on
    failure; every call site here checks for ``None`` and returns its own default.
    """
    return _shared_wordnet_module()


@lru_cache(maxsize=100_000)
def _candidate_wordnet_terms(text: str) -> tuple[str, ...]:
    tokens = [
        _light_stem(token.lower())
        for token in _word_tokens(text)
        if len(token) >= 3 and token.lower() not in _query_stopwords()
    ]
    out: list[str] = []
    seen: set[str] = set()
    for width in (3, 2, 1):
        for index in range(0, max(0, len(tokens) - width + 1)):
            phrase_tokens = tokens[index : index + width]
            if not phrase_tokens or any(len(token) < 3 for token in phrase_tokens):
                continue
            term = "_".join(phrase_tokens)
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
    return tuple(out)


_ROLE_HEAD_LEXNAMES = frozenset({"noun.person", "noun.group"})
_ROLE_MODIFIER_LEXNAMES = frozenset({"noun.act", "noun.communication", "noun.cognition"})
_ROLE_ACTION_VERB_LEXNAMES = frozenset({"verb.creation", "verb.social"})


@lru_cache(maxsize=8192)
def _wordnet_compound_role_action_terms(text: str, *, max_terms: int = 12) -> tuple[str, ...]:
    limit = max(0, int(max_terms))
    if limit <= 0:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = _normalize_wordnet_lookup_term(value)
        if len(token) < 3 or token in seen or token in _query_stopwords():
            return
        if token in _generic_wordnet_hypernyms() or token in _generic_wordnet_type_slot_hypernyms():
            return
        seen.add(token)
        out.append(token)

    for compound in _candidate_wordnet_terms(text):
        parts = [part for part in compound.split("_") if part]
        if len(parts) < 2:
            continue
        for index in range(1, len(parts)):
            modifier = parts[index - 1]
            head = parts[index]
            for term in _wordnet_role_action_terms_for_pair(modifier, head):
                add(term)
                if len(out) >= limit:
                    return tuple(out[:limit])
    return tuple(out[:limit])


@lru_cache(maxsize=8192)
def _wordnet_role_action_terms_for_pair(modifier: str, head: str) -> tuple[str, ...]:
    modifier = _normalize_wordnet_lookup_term(modifier)
    head = _normalize_wordnet_lookup_term(head)
    if not modifier or not head:
        return ()
    if not (_wordnet_term_lexnames(modifier) & _ROLE_MODIFIER_LEXNAMES):
        return ()
    if not (_wordnet_term_lexnames(head) & _ROLE_HEAD_LEXNAMES):
        return ()
    wn = _wordnet()
    if wn is None:
        return ()

    modifier_stems = _wordnet_surface_stems(modifier)
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = _normalize_wordnet_lookup_term(value)
        if len(token) < 3 or token in seen or token in _query_stopwords():
            return
        if token in _generic_wordnet_hypernyms() or token in _generic_wordnet_type_slot_hypernyms():
            return
        seen.add(token)
        out.append(token)

    def add_term_family(value: str, *, include_hypernyms: bool = True) -> None:
        add(value)
        for verb in _wordnet_derivationally_related_verbs(value):
            if _focused_role_action_verb(verb):
                add(verb)
        if not include_hypernyms:
            return
        for hypernym in _wordnet_direct_hypernyms(value):
            add(hypernym)
            for verb in _wordnet_derivationally_related_verbs(hypernym):
                if _focused_role_action_verb(verb):
                    add(verb)

    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(head):
            for synset in _wordnet_noun_synsets(wn, variant):
                if _role_synset_matches_modifier(synset, modifier_stems):
                    for lemma in synset.lemma_names()[:4]:
                        add_term_family(lemma)
                    for hypernym in synset.hypernyms()[:3]:
                        for lemma in hypernym.lemma_names()[:3]:
                            add_term_family(lemma, include_hypernyms=False)
                for hyponym in synset.hyponyms()[:64]:
                    if str(hyponym.lexname()) not in _ROLE_HEAD_LEXNAMES:
                        continue
                    if not _role_synset_matches_modifier(hyponym, modifier_stems):
                        continue
                    for lemma in hyponym.lemma_names()[:4]:
                        add_term_family(lemma)
                    for hypernym in hyponym.hypernyms()[:3]:
                        for lemma in hypernym.lemma_names()[:3]:
                            add_term_family(lemma, include_hypernyms=False)
    return tuple(out[:16])


def _role_synset_matches_modifier(synset: Any, modifier_stems: frozenset[str]) -> bool:
    if not modifier_stems:
        return False
    surface = " ".join(
        [
            str(synset.name()).replace("_", " "),
            str(synset.definition()),
            " ".join(str(item) for item in synset.lemma_names()),
            " ".join(str(item) for item in synset.examples()),
        ]
    )
    return bool(_wordnet_surface_stems(surface) & modifier_stems)


def _focused_role_action_verb(term: str) -> bool:
    lexnames = _wordnet_verb_lexnames(term)
    return bool(lexnames and lexnames <= _ROLE_ACTION_VERB_LEXNAMES)


def _wordnet_surface_stems(text: str) -> frozenset[str]:
    stems: set[str] = set()
    for token in _word_tokens(str(text or "").replace("_", " ")):
        lowered = token.lower()
        if len(lowered) < 3 or lowered in _query_stopwords():
            continue
        stemmed = _light_stem(lowered)
        if len(stemmed) >= 3 and stemmed not in _query_stopwords():
            stems.add(stemmed)
    for variant in _wordnet_lookup_variants(text):
        for part in variant.replace("_", " ").split():
            if len(part) >= 3 and part not in _query_stopwords():
                stems.add(_light_stem(part))
    return frozenset(stems)


@lru_cache(maxsize=4096)
def _wordnet_taxonomy_profile(term: str) -> tuple[frozenset[str], frozenset[str], float] | None:
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return None
    wn = _wordnet()
    if wn is None:
        return None
    with _WORDNET_LOCK:
        synsets = _wordnet_noun_synsets(wn, normalized)
        if not synsets:
            return None
        animal_synsets = [synset for synset in synsets if str(synset.lexname()) == "noun.animal"]
        first_is_animal = str(synsets[0].lexname()) == "noun.animal"
        if animal_synsets and (first_is_animal or "_" in normalized):
            synsets = animal_synsets
        else:
            synsets = [synset for synset in synsets[:3] if str(synset.lexname()) != "noun.animal"] or synsets[:1]
        synset_ids: set[str] = set()
        hypernym_names: set[str] = set()
        max_depth = 0
        for synset in synsets:
            synset_ids.add(f"{synset.pos()}:{synset.offset()}")
            max_depth = max(max_depth, int(synset.min_depth()))
            for hypernym in synset.closure(lambda item: item.hypernyms()):
                depth = int(hypernym.min_depth())
                if depth < 5:
                    continue
                max_depth = max(max_depth, depth)
                for lemma in hypernym.lemma_names():
                    name = lemma.lower()
                    if name in _generic_wordnet_hypernyms():
                        continue
                    if len(name) >= 3:
                        hypernym_names.add(name)
    return frozenset(synset_ids), frozenset(hypernym_names), float(max_depth)


@lru_cache(maxsize=4096)
def _wordnet_taxonomy_metadata_for_text(text: str, *, max_terms: int = 12) -> dict[str, list[str]]:
    wn = _wordnet()
    if wn is None:
        return {"terms": [], "hypernyms": [], "lexnames": []}
    term_counts: Counter[str] = Counter()
    hypernym_counts: Counter[str] = Counter()
    lexname_counts: Counter[str] = Counter()
    for term in _candidate_wordnet_taxonomy_terms(text)[:36]:
        profile = _wordnet_taxonomy_profile(term)
        if profile is None:
            continue
        synsets, hypernyms, _depth = profile
        if not synsets and not hypernyms:
            continue
        lexnames = _wordnet_term_lexnames(term)
        if "noun.quantity" in lexnames:
            continue
        display_term = term.replace("_", " ")
        term_counts[display_term] += 1
        for hypernym in _wordnet_direct_hypernyms(term) or tuple(hypernyms):
            clean_hypernym = hypernym.replace("_", " ")
            if len(clean_hypernym) >= 3:
                hypernym_counts[clean_hypernym] += 1
        for lexname in lexnames:
            clean_lexname = lexname.replace(".", ":")
            if clean_lexname and not clean_lexname.endswith(":Tops"):
                lexname_counts[clean_lexname] += 1
    limit = max(1, int(max_terms))
    return {
        "terms": _rank_taxonomy_terms(term_counts, limit=limit),
        "hypernyms": _rank_taxonomy_terms(hypernym_counts, limit=limit),
        "lexnames": _rank_taxonomy_terms(lexname_counts, limit=min(limit, 8)),
    }


def _rank_taxonomy_terms(counts: Counter[str], *, limit: int) -> list[str]:
    return [
        term
        for term, _count in sorted(
            counts.items(),
            key=lambda item: (item[1], len(item[0].split()), len(item[0]), item[0]),
            reverse=True,
        )[:limit]
        if term
    ]


@lru_cache(maxsize=4096)
def _candidate_wordnet_taxonomy_terms(text: str) -> tuple[str, ...]:
    doc = _doc(text)
    out: list[str] = []
    if doc is not None:
        try:
            for chunk in getattr(doc, "noun_chunks", []):
                phrase = _normalize_wordnet_lookup_term(
                    str(getattr(chunk, "lemma_", "") or getattr(chunk, "text", "") or "")
                )
                if phrase and len(phrase.replace("_", "")) >= 3:
                    out.append(phrase)
        except Exception as exc:
            # KEEP -- defensive guard around iterating an
            # already-successfully-parsed spaCy doc's noun_chunks; the
            # regex-tokenize fallback below (`if not out:`) is the same
            # degrade path already used when `doc is None`.
            logger.warning("wordnet noun_chunks extraction failed, falling back to regex tokens: %s", exc)
        try:
            for token in doc:
                if (
                    getattr(token, "is_space", False)
                    or getattr(token, "is_punct", False)
                    or getattr(token, "is_stop", False)
                ):
                    continue
                if str(getattr(token, "pos_", "") or "") not in {"NOUN", "PROPN"}:
                    continue
                lemma = _normalize_wordnet_lookup_term(
                    str(getattr(token, "lemma_", "") or getattr(token, "text", "") or "")
                )
                if lemma and len(lemma) >= 3 and lemma not in _query_stopwords():
                    out.append(lemma)
        except Exception as exc:
            # same class as the noun_chunks guard above --
            # defensive iteration guard with the identical regex fallback
            # below.
            logger.warning("wordnet token-lemma extraction failed, falling back to regex tokens: %s", exc)
    if not out:
        out.extend(
            _normalize_wordnet_lookup_term(token)
            for token in _word_tokens(text)
            if len(token) >= 3 and token.lower() not in _query_stopwords()
        )
    expanded: list[str] = []
    seen: set[str] = set()
    for term in out:
        variants = _wordnet_lookup_variants(term)
        canonical = next((variant for variant in variants[1:] if _wordnet_taxonomy_profile(variant) is not None), None)
        if canonical is None:
            canonical = next((variant for variant in variants if _wordnet_taxonomy_profile(variant) is not None), None)
        if canonical is None:
            canonical = variants[0] if variants else term
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        expanded.append(canonical)
    return tuple(expanded)


@lru_cache(maxsize=8192)
def _wordnet_direct_hypernyms(term: str) -> tuple[str, ...]:
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ()
    wn = _wordnet()
    if wn is None:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    with _WORDNET_LOCK:
        for synset in _wordnet_noun_synsets(wn, normalized)[:2]:
            for hypernym in synset.hypernyms()[:3]:
                if str(hypernym.lexname()) == "noun.Tops":
                    continue
                if int(hypernym.min_depth()) < 4:
                    continue
                for lemma in hypernym.lemma_names()[:2]:
                    name = lemma.lower()
                    if name in _generic_wordnet_hypernyms() or len(name) < 3 or name in seen:
                        continue
                    seen.add(name)
                    out.append(name)
    return tuple(out)


def _generic_wordnet_hypernyms() -> frozenset[str]:
    return frozenset(
        {
            "animate_being",
            "beast",
            "being",
            "brute",
            "causal_agent",
            "creature",
            "entity",
            "living_thing",
            "object",
            "organism",
            "physical_entity",
            "whole",
        }
    )


@lru_cache(maxsize=4096)
def _wordnet_term_lexnames(term: str) -> frozenset[str]:
    wn = _wordnet()
    if wn is None:
        return frozenset()
    lexnames: set[str] = set()
    with _WORDNET_LOCK:
        for synset in _wordnet_noun_synsets(wn, _normalize_wordnet_lookup_term(term)):
            lexnames.add(str(synset.lexname()))
    return frozenset(lexnames)


@lru_cache(maxsize=4096)
def _wordnet_noun_synset_shapes(term: str) -> tuple[tuple[str, int], ...]:
    wn = _wordnet()
    if wn is None:
        return ()
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ()
    # LOUD -- `wn` is already confirmed loaded (via
    # `wordnet_ready()`'s single seam above); a `.synsets()` call failing on
    # ordinary input here is a real bug, not an availability condition
    # .
    out: list[tuple[str, int]] = []
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(normalized):
            for synset in wn.synsets(variant, pos=wn.NOUN)[:8]:
                out.append((str(synset.lexname()), int(synset.min_depth())))
            if out:
                break
    return tuple(out)


@lru_cache(maxsize=4096)
def _wordnet_noun_sense_lexnames(term: str) -> tuple[str, ...]:
    """Every noun sense's lexname, in WordNet order (no sense cap); ``()`` when WordNet is unavailable or the term unknown."""
    wn = _wordnet()
    if wn is None:
        return ()
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ()
    out: list[str] = []
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(normalized):
            out.extend(str(synset.lexname()) for synset in wn.synsets(variant, pos=wn.NOUN))
            if out:
                break
    return tuple(out)


@lru_cache(maxsize=4096)
def _wordnet_first_noun_sense_closure(term: str) -> frozenset[str]:
    """The first noun sense's synset name and every hypernym above it ("sort" -> ``kind.n.01``, ...); empty when
    WordNet is unavailable or the term has no noun sense."""
    wn = _wordnet()
    if wn is None:
        return frozenset()
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return frozenset()
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(normalized):
            senses = wn.synsets(variant, pos=wn.NOUN)
            if senses:
                first = senses[0]
                return frozenset({first.name(), *(h.name() for h in first.closure(lambda s: s.hypernyms()))})
    return frozenset()


@lru_cache(maxsize=4096)
def _wordnet_noun_sense_closures(term: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Every noun sense of ``term`` (no sense cap) as ``(synset name, the synset name plus every hypernym and
    instance-hypernym above it)``; ``()`` when WordNet is unavailable or the term has no noun sense."""
    wn = _wordnet()
    if wn is None:
        return ()
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ()
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(normalized):
            senses = wn.synsets(variant, pos=wn.NOUN)
            if senses:
                return tuple(
                    (
                        sense.name(),
                        frozenset(
                            {
                                sense.name(),
                                *(h.name() for h in sense.closure(lambda s: [*s.hypernyms(), *s.instance_hypernyms()])),
                            }
                        ),
                    )
                    for sense in senses
                )
    return ()


def _wordnet_noun_senses_under(term: str, concept: str) -> bool | None:
    """Whether ANY noun sense of ``term`` lies under ANY noun sense of ``concept`` (hypernym and instance-hypernym
    closure, a sense under itself). ``None`` when either has no noun sense (or WordNet is unavailable): unknown."""
    sense_closures = _wordnet_noun_sense_closures(term)
    concept_senses = {name for name, _ in _wordnet_noun_sense_closures(concept)}
    if not sense_closures or not concept_senses:
        return None
    return any(closure & concept_senses for _, closure in sense_closures)


@lru_cache(maxsize=4096)
def _wordnet_verb_lexnames(term: str) -> frozenset[str]:
    wn = _wordnet()
    if wn is None:
        return frozenset()
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return frozenset()
    # LOUD -- see `_wordnet_noun_synset_shapes` above.
    lexnames: set[str] = set()
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(normalized):
            for synset in wn.synsets(variant, pos=wn.VERB):
                lexnames.add(str(synset.lexname()))
    return frozenset(lexnames)


@lru_cache(maxsize=4096)
def _wordnet_primary_verb_lexname(term: str) -> str:
    wn = _wordnet()
    if wn is None:
        return ""
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ""
    # LOUD -- see `_wordnet_noun_synset_shapes` above.
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(normalized):
            synsets = wn.synsets(variant, pos=wn.VERB)
            if synsets:
                return str(synsets[0].lexname())
    return ""


@lru_cache(maxsize=4096)
def _wordnet_verb_lemma(term: str) -> str:
    wn = _wordnet()
    if wn is None:
        return ""
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ""
    # LOUD -- see `_wordnet_noun_synset_shapes` above.
    with _WORDNET_LOCK:
        lemma = wn.morphy(normalized, wn.VERB)
    return str(lemma or "")


@lru_cache(maxsize=4096)
def _wordnet_duration_unit_term(term: str) -> bool:
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return False
    wn = _wordnet()
    if wn is None:
        return False
    # LOUD (both handlers below) -- `wn` is already confirmed
    # loaded; a `.synsets()`/`.closure()` call failing on ordinary input is a
    # real bug, not an availability condition.
    with _WORDNET_LOCK:
        synsets = wn.synsets(normalized, pos=wn.NOUN)[:5]
    has_duration_unit = False
    with _WORDNET_LOCK:
        for synset in synsets:
            name = str(getattr(synset, "name", lambda: "")())
            hypernyms = {str(item.name()) for item in synset.closure(lambda node: node.hypernyms())}
            if name == "time_unit.n.01" or "time_unit.n.01" in hypernyms:
                has_duration_unit = True
            elif name in {"time_period.n.01", "period.n.03"} or "time_period.n.01" in hypernyms:
                calendar_months = {"calendar_month.n.01", "gregorian_calendar_month.n.01"}
                if name not in calendar_months and not calendar_months & hypernyms:
                    has_duration_unit = True
    return has_duration_unit


def _generic_wordnet_type_slot_hypernyms() -> frozenset[str]:
    return frozenset(
        {
            "abstraction",
            "act",
            "action",
            "activity",
            "attribute",
            "communication",
            "creation",
            "entity",
            "event",
            "group",
            "object",
            "physical_entity",
            "process",
            "psychological_feature",
            "relation",
            "state",
            "thing",
            "whole",
            "work",
        }
    )


def _normalize_wordnet_lookup_term(term: str) -> str:
    return re.sub(r"[^a-z_]+", "", str(term or "").lower().replace(" ", "_"))


def _wordnet_lookup_variants(term: str) -> tuple[str, ...]:
    normalized = _normalize_wordnet_lookup_term(term)
    if not normalized:
        return ()
    variants = [normalized]
    parts = normalized.split("_")
    restored_parts = [_restore_wordnet_stem(part) for part in parts]
    restored = "_".join(restored_parts)
    if restored and restored not in variants:
        variants.append(restored)
    if "_" in normalized:
        spaced = normalized.replace("_", " ")
        if spaced not in variants:
            variants.append(spaced)
        restored_spaced = restored.replace("_", " ")
        if restored_spaced and restored_spaced not in variants:
            variants.append(restored_spaced)
    return tuple(variants)


def _wordnet_derivationally_related_verbs(term: str) -> tuple[str, ...]:
    return _wordnet_derivationally_related_verbs_cached(_normalize_wordnet_lookup_term(term))


@lru_cache(maxsize=8192)
def _wordnet_derivationally_related_verbs_cached(term: str) -> tuple[str, ...]:
    wn = _wordnet()
    if wn is None or not term:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(term):
            for synset in wn.synsets(variant)[:8]:
                for lemma in synset.lemmas()[:12]:
                    for related in lemma.derivationally_related_forms()[:12]:
                        related_synset = related.synset()
                        if related_synset is None or related_synset.pos() != "v":
                            continue
                        name = _light_stem(str(related.name() or "").lower().replace("_", " "))
                        verb = wn.morphy(name, wn.VERB) or name
                        verb = _light_stem(str(verb or "").lower())
                        if len(verb) >= 3 and verb not in seen and verb not in _query_stopwords():
                            seen.add(verb)
                            out.append(verb)
    return tuple(out[:8])


def _wordnet_derivationally_related_nouns(term: str) -> tuple[str, ...]:
    return _wordnet_derivationally_related_nouns_cached(_normalize_wordnet_lookup_term(term))


@lru_cache(maxsize=8192)
def _wordnet_derivationally_related_nouns_cached(term: str) -> tuple[str, ...]:
    wn = _wordnet()
    if wn is None or not term:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        noun = _normalize_wordnet_lookup_term(str(value or "").replace("_", " "))
        if len(noun) < 3 or noun in seen or noun in _query_stopwords():
            return
        seen.add(noun)
        out.append(noun)

    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(term):
            for synset in wn.synsets(variant)[:8]:
                for lemma in synset.lemmas()[:12]:
                    for related in lemma.derivationally_related_forms()[:12]:
                        related_synset = related.synset()
                        if related_synset is None or related_synset.pos() != "n":
                            continue
                        add(str(related.name() or ""))
    return tuple(out[:8])


def _wordnet_adjective_nominal_terms(term: str) -> tuple[str, ...]:
    return _wordnet_adjective_nominal_terms_cached(_normalize_wordnet_lookup_term(term))


@lru_cache(maxsize=8192)
def _wordnet_adjective_nominal_terms_cached(term: str) -> tuple[str, ...]:
    wn = _wordnet()
    if wn is None or not term:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        noun = _normalize_wordnet_lookup_term(str(value or "").replace("_", " "))
        if len(noun) < 3 or noun in seen or noun in _query_stopwords():
            return
        if noun in _generic_wordnet_hypernyms() or noun in _generic_wordnet_type_slot_hypernyms():
            return
        seen.add(noun)
        out.append(noun)

    for noun in _wordnet_derivationally_related_nouns(term):
        add(noun)

    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(term):
            for synset in wn.synsets(variant, pos=wn.ADJ)[:2]:
                definition_text = str(synset.definition() or "")
                for raw in _word_tokens(definition_text):
                    candidate = _normalize_wordnet_lookup_term(raw)
                    if _wordnet_term_lexnames(candidate):
                        add(candidate)
    return tuple(out[:12])


def _restore_wordnet_stem(term: str) -> str:
    if len(term) > 4 and term.endswith("i"):
        return f"{term[:-1]}y"
    return term


def _wordnet_noun_synsets(wn: Any, term: str) -> list[Any]:
    synsets: list[Any] = []
    with _WORDNET_LOCK:
        for variant in _wordnet_lookup_variants(term):
            synsets = list(wn.synsets(variant, pos=wn.NOUN))
            if synsets:
                break
    if not synsets:
        return []
    return synsets[:4]


def _word_tokens(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[A-Za-z][A-Za-z0-9']+", text or "")]


def _query_stopwords() -> frozenset[str]:
    return _english_stopwords()  # the wh words are stopwords


def _light_stem(token: str) -> str:
    if token.endswith("'s") and len(token) > 2:
        token = token[:-2]
    stemmer = _porter_stemmer()
    if stemmer is not None:
        return str(stemmer.stem(token))
    return token


@lru_cache(maxsize=1)
def _porter_stemmer() -> Any | None:
    """nltk PorterStemmer LOAD guard -- optional-dependency resource, not a
    stem call (mirrors the spaCy-model-load / wordnet-corpus
    KEEP class; `_light_stem` returns the token unstemmed for exactly this case)."""
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer()
    except Exception as exc:
        logger.warning("nltk PorterStemmer unavailable, tokens stay unstemmed: %s", exc)
        return None


def _clean_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


@lru_cache(maxsize=8192)
def _doc(text: str) -> Any | None:
    nlp = _load_nlp()
    if nlp is None:
        return None
    try:
        return nlp(text)
    except Exception as exc:
        # KEEP -- `text` is arbitrary conversation content,
        # and this module's ENTIRE architecture already treats `_doc(text) is
        # None` as an ordinary, non-fatal degrade signal (every caller has a
        # regex-tokenize fallback for it -- the same one used when spaCy is
        # not installed at all). Making a parse-failure on some specific text
        # crash here would be a narrower, silent behavior change on top of an
        # already-designed contract, not a bug surfaced.
        logger.warning("spaCy failed to parse a statement/query text, degrading to regex tokens: %s", exc)
        return None


@lru_cache(maxsize=1)
def _load_nlp() -> Any | None:
    # KEEP -- spaCy MODEL LOAD, not a parse call; the model
    # is a separately-downloaded resource, genuinely absent in some
    # deployments.
    try:
        return load_spacy_model()
    except Exception as exc:
        logger.warning("spaCy model unavailable, wordnet semantics degrade to regex-only tokenization: %s", exc)
        return None
