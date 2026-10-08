"""Exact expected-nDCG regret for finite stochastic ranking policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class MetricAlignedRankingError(ValueError):
    """Raised when a ranking-risk object is not mathematically well-defined."""


def _weights(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for index, raw in enumerate(values):
        if isinstance(raw, bool):
            raise MetricAlignedRankingError(
                f"position_weights[{index}] must be numeric"
            )
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise MetricAlignedRankingError(
                f"position_weights[{index}] must be finite and positive"
            )
        if result and value > result[-1] + 1e-15:
            raise MetricAlignedRankingError("position_weights must be non-increasing")
        result.append(value)
    if not result:
        raise MetricAlignedRankingError("position_weights must be non-empty")
    return tuple(result)


def ranking_exposure(
    distribution: Mapping[tuple[int, ...], float],
    *,
    item_count: int,
    position_weights: Sequence[float],
) -> tuple[float, ...]:
    """Compute expected item exposure under an explicit ranking mixture."""

    if not distribution:
        raise MetricAlignedRankingError("ranking distribution must be non-empty")
    weights = _weights(position_weights)
    if (
        isinstance(item_count, bool)
        or not isinstance(item_count, int)
        or item_count < 1
    ):
        raise MetricAlignedRankingError("item_count must be a positive integer")
    exposure = [0.0] * item_count
    total = 0.0
    for ranking, raw_probability in distribution.items():
        if len(ranking) != len(weights) or len(set(ranking)) != len(ranking):
            raise MetricAlignedRankingError(
                "each ranking must match the cutoff and contain unique items"
            )
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise MetricAlignedRankingError(
                "ranking probabilities must be finite and non-negative"
            )
        total += probability
        for position, item in enumerate(ranking):
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < item_count
            ):
                raise MetricAlignedRankingError("ranking item is out of range")
            exposure[item] += probability * weights[position]
    if abs(total - 1.0) > 1e-9:
        raise MetricAlignedRankingError("ranking probabilities must sum to one")
    return tuple(exposure)


@dataclass(frozen=True, slots=True)
class WorstCaseNdcgRegret:
    """Exact label-free regret certificate for an expected ranking policy."""

    regret: float
    relevant_count: int
    adversarial_relevant_items: tuple[int, ...]
    exposure_difference: tuple[float, ...]
    exact_for_binary_relevance: bool = True
    exact_for_arbitrary_nonnegative_graded_gains: bool = True
    label_access_required: bool = False
    expected_policy_only: bool = True


def worst_case_ndcg_regret(
    target_exposure: Sequence[float],
    candidate_exposure: Sequence[float],
    *,
    position_weights: Sequence[float],
) -> WorstCaseNdcgRegret:
    """Return the exact worst-case expected-nDCG loss from target to candidate."""

    if not target_exposure or len(target_exposure) != len(candidate_exposure):
        raise MetricAlignedRankingError(
            "exposure vectors must have equal non-zero length"
        )
    differences: list[float] = []
    target_total = 0.0
    candidate_total = 0.0
    for index, (raw_target, raw_candidate) in enumerate(
        zip(target_exposure, candidate_exposure)
    ):
        if isinstance(raw_target, bool) or isinstance(raw_candidate, bool):
            raise MetricAlignedRankingError(f"exposure[{index}] must be numeric")
        target = float(raw_target)
        candidate = float(raw_candidate)
        if (
            not math.isfinite(target)
            or not math.isfinite(candidate)
            or target < 0.0
            or candidate < 0.0
        ):
            raise MetricAlignedRankingError(
                f"exposure[{index}] must be finite and non-negative"
            )
        target_total += target
        candidate_total += candidate
        differences.append(target - candidate)

    weights = _weights(position_weights)
    expected_mass = math.fsum(weights)
    if (
        abs(target_total - expected_mass) > 1e-9
        or abs(candidate_total - expected_mass) > 1e-9
    ):
        raise MetricAlignedRankingError(
            "each exposure vector must sum to the position-weight total"
        )

    order = sorted(
        range(len(differences)),
        key=lambda item: (-differences[item], item),
    )
    numerator = 0.0
    ideal = 0.0
    best = -math.inf
    best_count = 0
    best_items: tuple[int, ...] = ()
    for count, item in enumerate(order, start=1):
        numerator += differences[item]
        if count <= len(weights):
            ideal += weights[count - 1]
        regret = numerator / ideal
        if regret > best:
            best = regret
            best_count = count
            best_items = tuple(sorted(order[:count]))
    return WorstCaseNdcgRegret(
        regret=max(0.0, best),
        relevant_count=best_count,
        adversarial_relevant_items=best_items,
        exposure_difference=tuple(differences),
    )


def expected_binary_ndcg(
    exposure: Sequence[float],
    relevant_items: Sequence[int],
    *,
    position_weights: Sequence[float],
) -> float:
    """Evaluate expected nDCG for an exposure vector and binary relevance."""

    if not exposure:
        raise MetricAlignedRankingError("exposure must be non-empty")
    relevant = tuple(relevant_items)
    if not relevant or len(set(relevant)) != len(relevant):
        raise MetricAlignedRankingError("relevant_items must be non-empty and unique")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item < len(exposure)
        for item in relevant
    ):
        raise MetricAlignedRankingError("relevant item is out of range")
    weights = _weights(position_weights)
    ideal = math.fsum(weights[: min(len(relevant), len(weights))])
    return math.fsum(float(exposure[item]) for item in relevant) / ideal


# Transitional function alias for theorem notation used in the proof tests.
worst_case_binary_ndcg_regret = worst_case_ndcg_regret
