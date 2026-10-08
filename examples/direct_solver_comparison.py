"""Compare the public policy interface with explicit lower-level assembly.

The hand assembly assumes the supplied fixture has already been validated.
It is an explanatory recipe, not another supported parser or a time-saving
measurement. Both paths use the same optimizer and verifier.
"""

import argparse
import json
import math
from pathlib import Path

from metric_aligned_ranking.partial_coverage_optimizer import (
    optimize_robust_slot_policy_compact,
)
from metric_aligned_ranking.policy_io import (
    SCHEMA,
    loads_policy_json,
    optimize_policy,
    verify_policy,
)


def assemble_direct(context, scores):
    target = context["target_ranking"]
    index = {item: i for i, item in enumerate(target)}
    result = optimize_robust_slot_policy_compact(
        tuple(range(len(target))),
        supported_items=tuple(index[item] for item in context["supported_items"]),
        source_scores={index[item]: score for item, score in scores.items()},
        position_weights=context["position_weights"],
        risk_budget=context["risk_budget"],
    )
    return {
        "schema": SCHEMA,
        "context": context,
        "mixture": [
            {
                "probability": entry.probability,
                "ranking": [target[item] for item in entry.ranking],
            }
            for entry in result.ranking_mixture
        ],
    }


def compare(context, scores):
    reports = {}
    for name, policy in (
        ("public", optimize_policy(context, source_scores=scores)),
        ("direct", assemble_direct(context, scores)),
    ):
        # Check what a recipient receives, not just the in-memory result.
        received = loads_policy_json(json.dumps(policy, allow_nan=False))
        report = verify_policy(received, expected_context=context)
        reports[name] = {
            "regret": report["regret"],
            "mixture_size": report["mixture_size"],
            "source_utility": math.fsum(
                scores.get(item, 0) * exposure
                for item, exposure in zip(
                    report["item_order"], report["candidate_exposure"]
                )
            ),
        }
    gap = abs(reports["public"]["source_utility"] - reports["direct"]["source_utility"])
    if gap > 1e-8:
        raise ValueError("public and direct source objectives differ by more than 1e-8")
    return {
        "valid": True,
        "synthetic_only": True,
        "same_core_solver": True,
        "objective_abs_difference": gap,
        "paths": reports,
    }


def self_check():
    cases = []
    for support in ([], ["zero"], ["zero", "source"]):
        for budget in (0.0, 0.1, 1.0):
            context = {
                "target_ranking": ["zero", "fixed", "source", "tail"],
                "supported_items": support,
                "position_weights": [1.0, 0.6],
                "risk_budget": budget,
            }
            scores = {item: {"zero": 0.0, "source": 2.0}[item] for item in support}
            result = compare(context, scores)
            cases.append(
                {
                    "support_size": len(support),
                    "budget": budget,
                    "objective_abs_difference": result["objective_abs_difference"],
                }
            )
    return {
        "valid": True,
        "synthetic_only": True,
        "case_count": len(cases),
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    examples = Path(__file__).resolve().parent
    parser.add_argument(
        "--context", type=Path, default=examples / "policy_context.json"
    )
    parser.add_argument("--scores", type=Path, default=examples / "policy_scores.json")
    parser.add_argument(
        "--self-check", action="store_true", help="run nine built-in synthetic settings"
    )
    args = parser.parse_args(argv)
    result = (
        self_check()
        if args.self_check
        else compare(
            loads_policy_json(args.context.read_text(encoding="utf-8-sig")),
            loads_policy_json(args.scores.read_text(encoding="utf-8-sig")),
        )
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
