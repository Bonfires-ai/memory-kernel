"""Native same-request analysis decoding and export.

Structural validity is not discourse authority. Image context is loader-attested,
not authenticated by the runtime grammar handle or caller-supplied digests.
Uninterned symbols keep exact printed names; duplicate native unit names refuse.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class DecodeError(ValueError):
    """Wire data is malformed, unsupported, incomplete, or over budget."""


def native_derender_nodes(nodes):
    """Copy parser nodes into the native de-renderer's particle-parent space.

    This mirrors fcg-propbank/fcg-components/de-render.lisp, not a semantic
    inference. Keep the original parser tree in every retained source envelope.
    """
    result = deepcopy(list(nodes))
    by_id = {node["node--id"]: node for node in result}
    for node in result:
        head = by_id.get(node.get("dependency--head"))
        if (
            node.get("node--type") == "leaf"
            and (node.get("lex--class", "").upper() == "RP" or node.get("dependency--label", "").lower() == "prt")
            and head is not None
            and head.get("node--type") == "leaf"
            and head.get("lex--class", "").upper().startswith("V")
            and head["end"] == node["start"]
        ):
            node["parent--constituent"] = head["node--id"]
    return result


class NativeNil(Enum):
    NIL = "native-nil"


NIL = NativeNil.NIL


@dataclass(frozen=True)
class NativeSymbol:
    package: str | None
    name: str


@dataclass(frozen=True)
class NativeNumber:
    """Exact supported Lisp numeric spelling; never converted to a Python float."""

    printed: str


@dataclass(frozen=True)
class NativeCons:
    car: NativeValue
    cdr: NativeValue


NativeValue = NativeNil | NativeSymbol | NativeNumber | NativeCons | str
_REAL = (
    r"[+-]?(?:[0-9]+/[1-9][0-9]*|"
    r"(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eEsSfFdDlL][+-]?[0-9]+)?|"
    r"[0-9]+[eEsSfFdDlL][+-]?[0-9]+|[0-9]+)"
)
_NUMBER = re.compile(rf"(?:{_REAL}|#C\({_REAL} {_REAL}\))", flags=re.ASCII)
_MAX_NODES = 200_000
_MAX_DEPTH = 200
_MAX_CHARS = 8_388_608


@dataclass
class _Decoder:
    max_nodes: int
    max_depth: int
    max_chars: int
    nodes: int = 0
    chars: int = 0

    def __post_init__(self) -> None:
        for value, ceiling, minimum in (
            (self.max_nodes, _MAX_NODES, 1),
            (self.max_depth, _MAX_DEPTH, 0),
            (self.max_chars, _MAX_CHARS, 0),
        ):
            if type(value) is not int or not minimum <= value <= ceiling:
                raise DecodeError("invalid decoder budget")

    def spend(self, *texts: str, nodes: int = 1) -> None:
        self.nodes += nodes
        self.chars += sum(len(text) for text in texts)
        if self.nodes > self.max_nodes or self.chars > self.max_chars:
            raise DecodeError("native tree exceeds resource budget")

    def decode(self, value: object, depth: int = 0) -> NativeValue:
        if depth > self.max_depth:
            raise DecodeError("native tree exceeds depth budget")
        self.spend()
        if type(value) is not dict or len(value) > 3:
            raise DecodeError("expected tagged native value")
        kind = value.get("kind")
        if type(kind) is not str:
            raise DecodeError("native kind must be a string")
        if kind == "nil" and set(value) == {"kind"}:
            return NIL
        if kind == "symbol" and set(value) == {"kind", "package", "name"}:
            package, name = value["package"], value["name"]
            if type(name) is not str or not (package is None or type(package) is str):
                raise DecodeError("invalid native symbol fields")
            self.spend(name, *(() if package is None else (package,)), nodes=0)
            return NativeSymbol(package, name)
        if kind == "string" and set(value) == {"kind", "value"}:
            if type(value["value"]) is not str:
                raise DecodeError("invalid native string")
            self.spend(value["value"], nodes=0)
            return value["value"]
        if kind == "number" and set(value) == {"kind", "printed"}:
            printed = value["printed"]
            if type(printed) is not str:
                raise DecodeError("invalid native number spelling")
            self.spend(printed, nodes=0)
            if len(printed) > 4096 or _NUMBER.fullmatch(printed) is None:
                raise DecodeError("unsupported native number spelling")
            return NativeNumber(printed)
        if kind == "cons" and set(value) == {"kind", "car", "cdr"}:
            return NativeCons(self.decode(value["car"], depth + 1), self.decode(value["cdr"], depth + 1))
        raise DecodeError("unknown, malformed, opaque or truncated native value")


def decode_tagged(
    value: object,
    *,
    max_nodes: int = _MAX_NODES,
    max_depth: int = _MAX_DEPTH,
    max_chars: int = _MAX_CHARS,
) -> NativeValue:
    """Decode one tagged value with bounded depth, visited nodes and text size."""
    return _Decoder(max_nodes, max_depth, max_chars).decode(value)


def proper_list(value: NativeValue) -> tuple[NativeValue, ...]:
    """Expose only a proper cons list; retain car values without interpretation."""
    items: list[NativeValue] = []
    visited: set[int] = set()
    while value is not NIL:
        if type(value) is not NativeCons:
            raise DecodeError("native list has a dotted or non-cons tail")
        if id(value) in visited or len(items) >= _MAX_NODES:
            raise DecodeError("native list is cyclic or over budget")
        visited.add(id(value))
        items.append(value.car)
        value = value.cdr
    return tuple(items)


def decode_snapshot(
    snapshot: object,
    *,
    max_nodes: int = _MAX_NODES,
    max_depth: int = _MAX_DEPTH,
    max_chars: int = _MAX_CHARS,
) -> dict[str, object]:
    """Decode probe envelope slot 3, sharing one budget across all native fields.

    JSON null in metadata collections remains None, distinct from tagged NIL.
    Accessor failures and nonzero limitation counters refuse the whole snapshot.
    Unknown metadata fields refuse instead of acquiring implied authority.
    """
    decoder = _Decoder(max_nodes, max_depth, max_chars)
    expected = {
        "node-state",
        "node-type",
        "meaning",
        "applications",
        "final-units",
        "non-list-units",
        "opaque-count",
        "truncated-count",
    }
    if type(snapshot) is not dict or len(snapshot) != len(expected) or set(snapshot) != expected:
        raise DecodeError("invalid native snapshot fields")
    for name in ("opaque-count", "truncated-count"):
        if type(snapshot[name]) is not int or snapshot[name] != 0:
            raise DecodeError("native snapshot has invalid or nonzero limitation count")
    if (
        type(snapshot["node-state"]) is not str
        or snapshot["node-state"]
        not in (
            "nil-node",
            "root-only",
            "applied",
        )
        or type(snapshot["node-type"]) is not str
    ):
        raise DecodeError("invalid native node metadata")
    decoder.spend(snapshot["node-state"], snapshot["node-type"])
    result: dict[str, object] = {
        "node-state": snapshot["node-state"],
        "node-type": snapshot["node-type"],
        "opaque-count": 0,
        "truncated-count": 0,
        "meaning": decoder.decode(snapshot["meaning"]),
    }
    for field in ("applications", "final-units"):
        records = snapshot[field]
        if records is None:
            result[field] = None
            continue
        if type(records) is not list or len(records) > max_nodes:
            raise DecodeError("invalid native record collection")
        decoded_records: list[dict[str, object]] = []
        previous_position = -1
        for record in records:
            decoder.spend()
            if type(record) is not dict:
                raise DecodeError("native record must be an object")
            if field == "applications":
                if len(record) != 4 or set(record) != {
                    "chain-position",
                    "construction",
                    "match-bindings",
                    "merge-bindings",
                }:
                    raise DecodeError("invalid native application fields")
                position = record["chain-position"]
                if type(position) is not int or position <= previous_position:
                    raise DecodeError("invalid native application chain position")
                previous_position = position
                construction = decoder.decode(record["construction"])
                if type(construction) is not NativeSymbol:
                    raise DecodeError("native construction name must be a symbol")
                decoded_records.append(
                    {
                        "chain-position": position,
                        "construction": construction,
                        "match-bindings": decoder.decode(record["match-bindings"]),
                        "merge-bindings": decoder.decode(record["merge-bindings"]),
                    }
                )
            else:
                if len(record) != 3 or set(record) != {"type", "unit-name", "raw"}:
                    raise DecodeError("invalid native unit fields")
                if type(record["type"]) is not str:
                    raise DecodeError("invalid native unit type")
                decoder.spend(record["type"], nodes=0)
                name = record["unit-name"]
                if type(name) is not dict or len(name) != 2 or set(name) != {"status", "value"}:
                    raise DecodeError("missing native unit name witness")
                if name["status"] != "returned":
                    raise DecodeError("native unit name accessor did not return")
                decoded_name = decoder.decode(name["value"])
                if type(decoded_name) is not NativeSymbol:
                    raise DecodeError("native unit name must be a symbol")
                decoded_records.append(
                    {
                        "type": record["type"],
                        "unit-name": {"status": "returned", "value": decoded_name},
                        "raw": decoder.decode(record["raw"]),
                    }
                )
        result[field] = decoded_records
    if (snapshot["node-state"] == "applied") != bool(result["applications"]):
        raise DecodeError("native node state disagrees with application records")
    non_list = snapshot["non-list-units"]
    if non_list is not None and snapshot["final-units"]:
        raise DecodeError("native units claim incompatible list and non-list forms")
    result["non-list-units"] = None if non_list is None else decoder.decode(non_list)
    return result


@dataclass(frozen=True)
class NativeUnit:
    name: NativeSymbol
    token_range: tuple[int, int]
    parser_node_id: int
    node_type: str
    parent: NativeSymbol | None
    features: tuple[tuple[NativeSymbol, NativeValue], ...]


@dataclass(frozen=True)
class NativeApplication:
    chain_position: int
    construction: NativeSymbol
    match_bindings: tuple[tuple[NativeSymbol, NativeValue], ...]
    merge_bindings: tuple[tuple[NativeSymbol, NativeValue], ...]
    bound_units: frozenset[NativeSymbol]


@dataclass(frozen=True)
class NativeAnalysis:
    """One selected native analysis, not a licensed discourse interpretation."""

    state: str
    meaning: tuple[tuple[NativeValue, ...], ...]
    units: tuple[NativeUnit, ...]
    applications: tuple[NativeApplication, ...]

    @property
    def units_by_name(self) -> Mapping[NativeSymbol, NativeUnit]:
        return MappingProxyType({unit.name: unit for unit in self.units})


def _integer(value: NativeValue) -> int:
    if type(value) is not NativeNumber or re.fullmatch(r"-?[0-9]+", value.printed) is None:
        raise DecodeError("native source coordinates must be integers")
    return int(value.printed)


def _bindings(value: NativeValue) -> tuple[tuple[NativeSymbol, NativeValue], ...]:
    out = []
    seen = set()
    for pair in proper_list(value):
        if type(pair) is not NativeCons or type(pair.car) is not NativeSymbol or not pair.car.name.startswith("?"):
            raise DecodeError("native application bindings require variable/value cons pairs")
        if pair.car in seen:
            raise DecodeError("native application repeats a binding variable")
        seen.add(pair.car)
        out.append((pair.car, pair.cdr))
    return tuple(out)


def validate_native_analysis(
    snapshot: object,
    *,
    nodes: Sequence[Mapping[str, object]],
    token_range: tuple[int, int],
    has_frames: bool = False,
) -> NativeAnalysis:
    """Validate exact unit features against the already-validated same-tree nodes.

    Membership uses complete exported symbols, never suffixes or surface text.
    A caller consuming a meaning row must additionally authorize its construction
    and verify every referenced source unit is in that application's bound_units.
    """
    decoded = decode_snapshot(snapshot)
    if decoded["non-list-units"] is not None:
        raise DecodeError("native analysis requires a structured final-unit list")
    rows = tuple(proper_list(row) for row in proper_list(decoded["meaning"]))
    if any(not row or type(row[0]) is not NativeSymbol for row in rows):
        raise DecodeError("native meaning requires predicate lists headed by symbols")
    units = []
    names: set[NativeSymbol] = set()
    parser_ids: set[int] = set()
    for record in decoded["final-units"] or ():
        raw = proper_list(record["raw"])
        name = record["unit-name"]["value"]
        if not raw or raw[0] != name or name in names:
            raise DecodeError("native unit identity is missing, duplicated or disagrees with accessor")
        names.add(name)
        features: dict[NativeSymbol, NativeValue] = {}
        for item in raw[1:]:
            feature = proper_list(item)
            if len(feature) != 2 or type(feature[0]) is not NativeSymbol or feature[0] in features:
                raise DecodeError("native unit features require unique exact key/value pairs")
            features[feature[0]] = feature[1]
        span = features.get(NativeSymbol("FCG-PROPBANK", "SPAN"), NIL)
        coordinates = proper_list(span)
        if len(coordinates) != 2:
            raise DecodeError("native unit requires one exact SPAN")
        start, end = (_integer(value) for value in coordinates)
        if not token_range[0] <= start < end <= token_range[1]:
            raise DecodeError("native unit span escapes its tree")
        kind = features.get(NativeSymbol("FCG-PROPBANK", "NODE-TYPE"))
        if kind not in (NativeSymbol("FCG-PROPBANK", "LEAF"), NativeSymbol("FCG-PROPBANK", "PHRASE")):
            raise DecodeError("native unit is not a supported source syntax node")
        node_type = kind.name.lower()
        candidates = [
            node
            for node in nodes
            if node.get("node--type") == node_type and node.get("start") == start and node.get("end") == end
        ]
        if len(candidates) != 1:
            raise DecodeError("native unit has no unique same-tree parser node")
        node = candidates[0]
        node_id = node.get("node--id")
        if type(node_id) is not int or node_id in parser_ids:
            raise DecodeError("native unit repeats a parser node")
        parser_ids.add(node_id)
        if features.get(NativeSymbol("COMMON-LISP", "STRING")) != node.get("string"):
            raise DecodeError("native unit string disagrees with its exact parser node")
        parent = features.get(NativeSymbol("UTILS", "PARENT"))
        if parent is not NIL and type(parent) is not NativeSymbol:
            raise DecodeError("native unit lacks a supported parent feature")
        units.append(
            NativeUnit(
                name, (start, end), node_id, node_type, None if parent is NIL else parent, tuple(features.items())
            )
        )
    unit_map = {unit.name: unit for unit in units}
    node_map = {node["node--id"]: node for node in nodes}
    for unit in units:
        parent = unit_map.get(unit.parent) if unit.parent is not None else None
        if unit.parent is not None and parent is None:
            raise DecodeError("native unit parent is outside this analysis")
        if node_map[unit.parser_node_id].get("parent--constituent") != (parent.parser_node_id if parent else None):
            raise DecodeError("native parent disagrees with the same-tree parser structure")
    applications = []
    for record in decoded["applications"] or ():
        match, merge = _bindings(record["match-bindings"]), _bindings(record["merge-bindings"])
        bound = frozenset(value for _, value in (*match, *merge) if type(value) is NativeSymbol and value in names)
        # An uninterned, non-variable binding value is a transient unit reference
        # in the supported native grammar, not a silently accepted foreign unit.
        if any(
            type(value) is NativeSymbol
            and value.package is None
            and not value.name.startswith("?")
            and value not in names
            for _, value in (*match, *merge)
        ):
            raise DecodeError("native application references a foreign transient unit")
        applications.append(NativeApplication(record["chain-position"], record["construction"], match, merge, bound))
    state = decoded["node-state"]
    if state == "nil-node" and (rows or units or applications or has_frames):
        raise DecodeError("nil native node cannot carry analysis evidence")
    if state != "nil-node" and not units:
        raise DecodeError("settled native node requires its source units")
    if state != "nil-node" and parser_ids != set(node_map):
        raise DecodeError("settled native node is missing source syntax units")
    if state == "root-only" and (rows or has_frames):
        raise DecodeError("root-only native analysis cannot acquire meaning or frames")
    return NativeAnalysis(state, rows, tuple(units), tuple(applications))


# Same-search walker proven by dqr001-native-discourse-proof-v3. No script import.
NATIVE_ANALYSIS_WALKER = r"""
(let ((visited 0) (opaque 0) (truncated 0))
 (labels
  ((wire (value &optional (depth 0))
     (incf visited)
     (cond
      ((or (> visited 200000) (> depth 200))
       (incf truncated) (list (cons :kind "truncated")))
      ((null value) (list (cons :kind "nil")))
      ((symbolp value)
       (list (cons :kind "symbol") (cons :name (symbol-name value))
             (cons :package (when (symbol-package value) (package-name (symbol-package value))))))
      ((stringp value) (list (cons :kind "string") (cons :value value)))
      ((numberp value)
       (list (cons :kind "number") (cons :printed (prin1-to-string value))))
      ((consp value)
       (list (cons :kind "cons") (cons :car (wire (car value) (1+ depth)))
             (cons :cdr (wire (cdr value) (1+ depth)))))
      (t
       (incf opaque)
       (list (cons :kind "opaque") (cons :type (prin1-to-string (type-of value)))
             (cons :printed (let ((*print-circle* t) (*print-length* 20) (*print-level* 6))
                              (prin1-to-string value)))))))
   (node-units (node)
     (when (and node (fcg::cipn-car node))
       (let ((cfs (fcg::car-resulting-cfs (fcg::cipn-car node))))
         (when cfs (fcg::left-pole-structure cfs)))))
   (unit-record (unit)
     (let* ((accessor (find-symbol "UNIT-NAME" :fcg))
            (name (when (and accessor (fboundp accessor))
                    (handler-case
                        (list (cons :status "returned") (cons :value (wire (funcall accessor unit))))
                      (error (e) (list (cons :status "accessor-error") (cons :error (princ-to-string e))))))))
       (list (cons :type (prin1-to-string (type-of unit)))
             (cons :unit-name (or name (list (cons :status "accessor-unavailable"))))
             (cons :raw (wire unit)))))
   (snapshot (meaning node)
     (let* ((chain (when node (reverse (cons node (fcg::all-parents node)))))
            (applications
             (loop for current in chain for ordinal from 0
                   for application = (fcg::cipn-car current)
                   when (and application (fcg::car-applied-cxn application))
                   collect
                    (list (cons :chain-position ordinal)
                          (cons :construction (wire (fcg::name (fcg::car-applied-cxn application))))
                          (cons :match-bindings (wire (fcg::car-match-bindings application)))
                          (cons :merge-bindings (wire (fcg::car-second-merge-bindings application))))))
            (units (node-units node))
            (unit-records (if (listp units) (mapcar #'unit-record units) nil))
            (raw-meaning (wire meaning))
            (raw-units (unless (listp units) (wire units))))
       (list (cons :node-state (cond ((null node) "nil-node") ((null applications) "root-only") (t "applied")))
             (cons :node-type (prin1-to-string (type-of node)))
             (cons :meaning raw-meaning)
             (cons :applications applications)
             (cons :final-units unit-records)
             (cons :non-list-units raw-units)
             (cons :opaque-count opaque) (cons :truncated-count truncated)))))
  __REQUEST__))
"""


def wrap_native_analysis_request(request: str) -> str:
    """Supply the native request with its local snapshot function, without another search."""
    return NATIVE_ANALYSIS_WALKER.replace("__REQUEST__", request)
