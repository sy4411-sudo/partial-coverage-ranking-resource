"""Exact graded-nDCG regret certificates for stochastic ranking policies.

The core observation is a finite layer-cake decomposition.  Every non-negative
gain vector is a non-negative sum of nested binary threshold vectors.  Both
the exposure-weighted DCG difference and the ideal DCG decompose over the same
thresholds.  Graded-nDCG regret is therefore a convex combination of binary
threshold regrets, so a binary relevance vector is extremal even when the
adversary may choose arbitrary non-negative graded gains.

It is part of the split metric-aligned ranking theory object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .ndcg_risk import (
    MetricAlignedRankingError,
    worst_case_binary_ndcg_regret,
)


@dataclass(frozen=True, slots=True)
class NdcgThresholdLayer:
    """One binary threshold in a non-negative graded gain vector."""

    gain_increment: float
    relevant_items: tuple[int, ...]
    ideal_binary_dcg: float
    exposure_difference: float
    binary_regret: float
    convex_weight: float


@dataclass(frozen=True, slots=True)
class GradedNdcgRegretCertificate:
    """Certificate that a graded regret is bounded by a binary adversary."""

    regret: float
    worst_case_regret: float
    upper_bound_gap: float
    ideal_graded_dcg: float
    layers: tuple[NdcgThresholdLayer, ...]
    binary_adversaries_are_extremal: bool = True
    accepts_arbitrary_nonnegative_gains: bool = True
    label_access_required_for_worst_case: bool = False


def _nonnegative_gains(
    gains: Sequence[float],
    *,
    size: int,
) -> tuple[float, ...]:
    if len(gains) != size:
        raise MetricAlignedRankingError(
            "gains must have the same length as the exposure vectors"
        )
    result: list[float] = []
    for index, value in enumerate(gains):
        if isinstance(value, bool):
            raise MetricAlignedRankingError(f"gains[{index}] must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise MetricAlignedRankingError(
                f"gains[{index}] must be finite and non-negative"
            )
        result.append(numeric)
    if not any(value > 0.0 for value in result):
        raise MetricAlignedRankingError("at least one gain must be positive")
    return tuple(result)


def graded_ndcg_regret_certificate(
    target_exposure: Sequence[float],
    candidate_exposure: Sequence[float],
    gains: Sequence[float],
    *,
    position_weights: Sequence[float],
) -> GradedNdcgRegretCertificate:
    """Bound one graded regret by the exact worst-case binary regret.

    The returned bound is exact over the class of all non-zero, non-negative
    gain vectors, including standard gain transforms such as ``2**rel - 1``.
    The supplied ``gains`` are used only to expose the threshold decomposition;
    computing the worst-case value itself remains label-free.
    """

    binary = worst_case_binary_ndcg_regret(
        target_exposure,
        candidate_exposure,
        position_weights=position_weights,
    )
    gain_vector = _nonnegative_gains(gains, size=len(target_exposure))
    weights = tuple(float(value) for value in position_weights)
    positive_levels = sorted({gain for gain in gain_vector if gain > 0.0})

    layer_data: list[tuple[float, tuple[int, ...], float, float, float]] = []
    previous_level = 0.0
    ideal_graded_dcg = 0.0
    graded_difference = 0.0
    for level in positive_levels:
        increment = level - previous_level
        relevant_items = tuple(
            index for index, gain in enumerate(gain_vector) if gain >= level
        )
        ideal_binary_dcg = math.fsum(weights[: min(len(relevant_items), len(weights))])
        exposure_difference = math.fsum(
            binary.exposure_difference[index] for index in relevant_items
        )
        binary_regret = exposure_difference / ideal_binary_dcg
        layer_data.append(
            (
                increment,
                relevant_items,
                ideal_binary_dcg,
                exposure_difference,
                binary_regret,
            )
        )
        ideal_graded_dcg += increment * ideal_binary_dcg
        graded_difference += increment * exposure_difference
        previous_level = level

    regret = graded_difference / ideal_graded_dcg
    layers = tuple(
        NdcgThresholdLayer(
            gain_increment=increment,
            relevant_items=relevant_items,
            ideal_binary_dcg=ideal_binary_dcg,
            exposure_difference=exposure_difference,
            binary_regret=binary_regret,
            convex_weight=increment * ideal_binary_dcg / ideal_graded_dcg,
        )
        for (
            increment,
            relevant_items,
            ideal_binary_dcg,
            exposure_difference,
            binary_regret,
        ) in layer_data
    )
    tolerance = 1e-12
    if regret > binary.regret + tolerance:
        raise AssertionError("graded regret exceeded its exact binary upper bound")
    if abs(math.fsum(layer.convex_weight for layer in layers) - 1.0) > tolerance:
        raise AssertionError("threshold layer weights do not form a convex combination")

    return GradedNdcgRegretCertificate(
        regret=regret,
        worst_case_regret=binary.regret,
        upper_bound_gap=binary.regret - regret,
        ideal_graded_dcg=ideal_graded_dcg,
        layers=layers,
    )


def expected_graded_ndcg(
    exposure: Sequence[float],
    gains: Sequence[float],
    *,
    position_weights: Sequence[float],
) -> float:
    """Evaluate expected nDCG for arbitrary non-negative gains."""

    worst_case_binary_ndcg_regret(
        exposure,
        exposure,
        position_weights=position_weights,
    )
    gain_vector = _nonnegative_gains(gains, size=len(exposure))
    sorted_gains = sorted(gain_vector, reverse=True)
    weights = tuple(float(value) for value in position_weights)
    ideal_dcg = math.fsum(gain * weight for gain, weight in zip(sorted_gains, weights))
    return (
        math.fsum(
            gain * float(item_exposure)
            for gain, item_exposure in zip(gain_vector, exposure)
        )
        / ideal_dcg
    )
