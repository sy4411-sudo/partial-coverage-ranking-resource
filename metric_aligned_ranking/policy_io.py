"""Portable policies with solver-independent expected-regret verification.

The trusted reference context must be supplied separately from the policy.
Feasibility does not establish optimality or empirical usefulness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

from .ndcg_risk import MetricAlignedRankingError, worst_case_ndcg_regret

SCHEMA = "metric-aligned-ranking.policy.v1"
PROBABILITY_TOLERANCE = 1e-12
REGRET_TOLERANCE = 1e-9


class PolicyFormatError(MetricAlignedRankingError):
    """The request or emitted policy is not valid for verification."""


class PolicyOptimizationError(RuntimeError):
    """The solver did not produce a policy; this is not proof of infeasibility."""


def _object(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise PolicyFormatError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyFormatError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise PolicyFormatError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise PolicyFormatError(f"{name} must be a finite number")
    return result


def _ids(value: object, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise PolicyFormatError(f"{name} must be a list of unique nonempty strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise PolicyFormatError(f"{name} must contain nonempty string IDs")
    if len(set(value)) != len(value):
        raise PolicyFormatError(f"{name} contains duplicate IDs")
    return list(value)


def validate_context(context: object) -> dict:
    """Return a validated copy of the reference ranking and regret constraints."""
    data = _object(
        context,
        {"target_ranking", "supported_items", "position_weights", "risk_budget"},
        "context",
    )
    target = _ids(data["target_ranking"], "target_ranking")
    support = _ids(data["supported_items"], "supported_items", allow_empty=True)
    if not set(support) <= set(target):
        raise PolicyFormatError("supported_items must be a subset of target_ranking")
    raw_weights = data["position_weights"]
    if not isinstance(raw_weights, list) or not 1 <= len(raw_weights) <= len(target):
        raise PolicyFormatError("position_weights must have between 1 and n entries")
    weights = [_number(w, "position weight") for w in raw_weights]
    if any(w <= 0 for w in weights) or any(
        left < right for left, right in pairwise(weights)
    ):
        raise PolicyFormatError("position_weights must be positive and non-increasing")
    # nDCG is scale-invariant in real arithmetic, not at subnormal float scales.
    # Keep normalization caller-owned so the trusted context is never rewritten.
    if weights[0] != 1.0:
        raise PolicyFormatError(
            "position_weights[0] must equal 1.0; normalize weights before "
            "constructing the trusted context"
        )
    try:
        if not math.isfinite(math.fsum(weights)):
            raise OverflowError
    except OverflowError as exc:
        raise PolicyFormatError("position-weight total must be finite") from exc
    budget = _number(data["risk_budget"], "risk_budget")
    if budget < 0:
        raise PolicyFormatError("risk_budget must be non-negative")
    return {
        "target_ranking": target,
        "supported_items": support,
        "position_weights": weights,
        "risk_budget": budget,
    }


def verify_policy(policy: object, *, expected_context: object) -> dict:
    """Recompute feasibility from emitted rankings, without importing SciPy.

    Probabilities are normalized by their sum, which must first be within
    1e-12 of one. Sampling must use these normalized probabilities, not the
    argmax ranking. The float certificate uses absolute tolerance 1e-9 and
    requires position_weights[0] == 1.0 to avoid arbitrary overall float scales.
    """
    expected = validate_context(expected_context)
    data = _object(policy, {"schema", "context", "mixture"}, "policy")
    if data["schema"] != SCHEMA:
        raise PolicyFormatError("unsupported policy schema")
    context = validate_context(data["context"])
    if context != expected:
        raise PolicyFormatError("policy context differs from expected context")
    mixture = data["mixture"]
    if not isinstance(mixture, list) or not mixture:
        raise PolicyFormatError("mixture must be a nonempty list")
    target = context["target_ranking"]
    target_set = set(target)
    support = set(context["supported_items"])
    checked = []
    for component in mixture:
        entry = _object(component, {"probability", "ranking"}, "mixture component")
        probability = _number(entry["probability"], "probability")
        if not 0 <= probability <= 1:
            raise PolicyFormatError("probability must lie in [0, 1]")
        ranking = _ids(entry["ranking"], "component ranking")
        if set(ranking) != target_set:
            raise PolicyFormatError("each ranking must permute all target IDs")
        if any(
            item not in support and ranking[position] != item
            for position, item in enumerate(target)
        ):
            raise PolicyFormatError("unsupported items must retain their target slots")
        checked.append((probability, ranking))
    mass = math.fsum(probability for probability, _ in checked)
    if abs(mass - 1) > PROBABILITY_TOLERANCE:
        raise PolicyFormatError(
            f"probabilities must sum to one within 1e-12; actual sum={mass:.17g}, "
            f"absolute error={abs(mass - 1):.17g}"
        )
    weights = context["position_weights"]
    index = {item: position for position, item in enumerate(target)}
    terms = [[] for _ in target]
    for probability, ranking in checked:
        for item, weight in zip(ranking, weights):
            terms[index[item]].append(probability / mass * weight)
    candidate_exposure = tuple(math.fsum(values) for values in terms)
    target_exposure = tuple(weights + [0.0] * (len(target) - len(weights)))
    certificate = worst_case_ndcg_regret(
        target_exposure, candidate_exposure, position_weights=weights
    )
    if certificate.regret > context["risk_budget"] + REGRET_TOLERANCE:
        raise PolicyFormatError(
            f"emitted policy exceeds expected regret budget: regret={certificate.regret:.17g}, "
            f"budget={context['risk_budget']:.17g}, tolerance={REGRET_TOLERANCE:.17g}"
        )
    return {
        "valid": True,
        "expected_policy_only": True,
        "regret": certificate.regret,
        "risk_budget": context["risk_budget"],
        "regret_tolerance": REGRET_TOLERANCE,
        "probability_sum_before_normalization": mass,
        "item_order": list(target),
        "supported_items": list(context["supported_items"]),
        "unsupported_items": [item for item in target if item not in support],
        "target_exposure": list(target_exposure),
        "candidate_exposure": list(candidate_exposure),
        "adversarial_relevant_items": [
            target[item] for item in certificate.adversarial_relevant_items
        ],
        "mixture_size": len(checked),
        "optimality_checked": False,
        "empirical_utility_checked": False,
    }


def optimize_policy(
    context: object, *, source_scores: Mapping[str, float], formulation: str = "compact"
) -> dict:
    """Solve with string IDs and verify the exported policy. Requires SciPy.

    Scores must exist exactly on supported_items. They are optimization
    inputs, not evidence of held-out relevance, and are not exported.
    """
    context = validate_context(context)
    if formulation not in {"compact", "cutting-plane"}:
        raise PolicyFormatError("formulation must be compact or cutting-plane")
    if not isinstance(source_scores, Mapping) or set(source_scores) != set(
        context["supported_items"]
    ):
        raise PolicyFormatError("source_scores keys must equal supported_items")
    index = {item: position for position, item in enumerate(context["target_ranking"])}
    scores = {
        index[item]: _number(score, "source score")
        for item, score in source_scores.items()
    }
    from . import partial_coverage_optimizer as optimizer

    solve = (
        optimizer.optimize_robust_slot_policy_compact
        if formulation == "compact"
        else optimizer.optimize_robust_slot_policy
    )
    try:
        result = solve(
            tuple(range(len(index))),
            supported_items=tuple(index[item] for item in context["supported_items"]),
            source_scores=scores,
            position_weights=context["position_weights"],
            risk_budget=context["risk_budget"],
        )
    except optimizer.RobustSlotRankingError as exc:
        raise PolicyOptimizationError(str(exc)) from exc
    target = context["target_ranking"]
    policy = {
        "schema": SCHEMA,
        "context": context,
        "mixture": [
            {
                "probability": c.probability,
                "ranking": [target[item] for item in c.ranking],
            }
            for c in result.ranking_mixture
        ],
    }
    verify_policy(policy, expected_context=context)
    return policy


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PolicyFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PolicyFormatError(f"non-finite JSON constant: {value}")


def loads_policy_json(text: str) -> object:
    """Parse JSON without duplicate-key overwrite or non-finite extensions."""
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError) as exc:
        raise PolicyFormatError(f"invalid policy JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("optimize", help="emit policy JSON to stdout")
    solve.add_argument("context", type=Path)
    solve.add_argument("scores", type=Path)
    solve.add_argument(
        "--formulation", choices=["compact", "cutting-plane"], default="compact"
    )
    verify = commands.add_parser("verify", help="check against a separate context")
    verify.add_argument("policy", type=Path)
    verify.add_argument("context", type=Path)
    args = parser.parse_args(argv)
    try:
        context = loads_policy_json(args.context.read_text(encoding="utf-8-sig"))
        if args.command == "optimize":
            scores = loads_policy_json(args.scores.read_text(encoding="utf-8-sig"))
            output = optimize_policy(
                context, source_scores=scores, formulation=args.formulation
            )
        else:
            policy = loads_policy_json(args.policy.read_text(encoding="utf-8-sig"))
            output = verify_policy(policy, expected_context=context)
        print(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False))
    except (
        MetricAlignedRankingError,
        PolicyOptimizationError,
        OSError,
        UnicodeError,
    ) as exc:
        print(f"policy error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
