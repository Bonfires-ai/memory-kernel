"""Bounded decimal rounding with unchanged probability and margin gates."""

from decimal import Decimal

POLICY = "bounded_decimal_rounding_v1"


def _precision_exponent(value: Decimal) -> int:
    """Decimal exponent of a provider value, ignoring binary float representation noise."""
    cleaned = value if value.is_zero() else value.quantize(Decimal(1).scaleb(-15)).normalize()
    exponent = cleaned.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError("provider distribution contains invalid probabilities")
    return exponent


def normalize_integrated_probabilities(raw, selected):
    """Mirror the recovery pilot's decimal bound; keep original bytes outside."""
    if (
        not raw
        or selected not in raw
        or any(isinstance(value, bool) or not isinstance(value, (int, Decimal)) for value in raw.values())
    ):
        raise ValueError("provider distribution contains invalid probabilities")
    values = {key: Decimal(value) for key, value in raw.items()}
    if any(not value.is_finite() or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("provider distribution contains invalid probabilities")
    total = sum(values.values())
    if abs(total - 1) <= Decimal("0.000001"):
        return {key: float(value) for key, value in values.items()}, None
    # A provider value's decimal precision bounds its rounding. Binary float noise in the provider's JSON
    # ("0.9400000000000001" for 0.94) is not real precision, so the exponent is read after rounding to 15
    # places. This only widens the bound for such noise; reported values and normalization are unchanged.
    bound = min(
        Decimal("0.02"),
        sum(
            Decimal(5).scaleb(_precision_exponent(value) - 1) if isinstance(value, Decimal) else Decimal(0)
            for value in raw.values()
        ),
    )
    if total <= 0 or abs(total - 1) > bound:
        raise ValueError("provider distribution exceeds bounded decimal rounding")

    def gates(distribution):
        represented = {key: float(value) for key, value in distribution.items()}
        probability = represented[selected]
        runner_up = max(value for key, value in represented.items() if key != selected)
        return {
            "argmax": probability >= runner_up,
            "probability_gate": probability >= 0.75,
            "margin_gate": probability - runner_up >= 0.20,
        }

    normalized = {key: value / total for key, value in values.items()}
    before, after = gates(values), gates(normalized)
    if before != after:
        raise ValueError("decimal normalization crosses an acceptance gate")
    probabilities = {key: float(value) for key, value in normalized.items()}
    return probabilities, {
        "policy": POLICY,
        "raw_total": str(total),
        "rounding_bound": str(bound),
        "raw_probabilities": {key: str(value) for key, value in values.items()},
        "normalized_probabilities": probabilities,
        "raw_gate": before,
        "normalized_gate": after,
    }
