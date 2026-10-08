"""Evaluate and sample received ranking mixtures without importing a solver.

Evaluation gains are a separate, complete candidate-universe input. This
example is not a production sampler audit or an estimator for missing labels.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from metric_aligned_ranking.policy_io import (
    PolicyFormatError,
    loads_policy_json,
    verify_policy,
)
from metric_aligned_ranking.run_adapter import verify_runs


def checked_gains(raw, requests):
    if not isinstance(raw, dict) or set(raw) != set(requests):
        raise ValueError("gain queries must exactly match expected queries")
    result = {}
    for qid, request in requests.items():
        values = raw[qid]
        if not isinstance(values, dict) or set(values) != set(
            request["context"]["target_ranking"]
        ):
            raise ValueError(f"query {qid!r}: gains must cover exactly all candidates")
        checked = {}
        for item, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PolicyFormatError("gains must be finite nonnegative numbers")
            try:
                gain = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("gains must be finite nonnegative numbers") from exc
            if not math.isfinite(gain) or gain < 0:
                raise ValueError("gains must be finite nonnegative numbers")
            checked[item] = gain
        # nDCG is scale invariant. Rescale before products to avoid overflow;
        # gains are already gains, not grades to exponentiate.
        scale = max(checked.values())
        result[qid] = {
            item: value / scale if scale else 0.0 for item, value in checked.items()
        }
    return result


def evaluate(batch, requests, raw_gains, *, draws=10000, seed=0):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PolicyFormatError("seed must be an integer")
    # Verify before consuming gains or drawing a ranking. The independently
    # supplied requests, never the received context, define the constraint.
    checked = verify_runs(batch, expected_requests=requests)
    gains = checked_gains(raw_gains, requests)
    rows = {}
    for qid in sorted(requests):
        context = requests[qid]["context"]
        policy = batch["policies"][qid]
        report = checked["reports"][qid]
        weights = context["position_weights"]
        ideal = math.fsum(
            gain * weight
            for gain, weight in zip(sorted(gains[qid].values(), reverse=True), weights)
        )

        def metric(
            ranking, query_gains=gains[qid], metric_weights=weights, denominator=ideal
        ):
            return (
                math.fsum(
                    query_gains[item] * w for item, w in zip(ranking, metric_weights)
                )
                / denominator
                if denominator
                else 0.0
            )

        components = policy["mixture"]
        mass = report["probability_sum_before_normalization"]
        probabilities = [entry["probability"] / mass for entry in components]
        metrics = [metric(entry["ranking"]) for entry in components]
        expected = math.fsum(p * value for p, value in zip(probabilities, metrics))
        by_exposure = (
            math.fsum(
                gains[qid][item] * exposure
                for item, exposure in zip(
                    report["item_order"], report["candidate_exposure"]
                )
            )
            / ideal
            if ideal
            else 0.0
        )
        if abs(expected - by_exposure) > 1e-12:
            raise ValueError("mixture and exposure evaluations disagree")
        # Query-specific streams are stable under input dictionary reordering.
        rng = random.Random(f"{seed}:{qid}")
        counts = Counter(
            rng.choices(range(len(components)), weights=probabilities, k=draws)
        )
        target = context["target_ranking"]
        support = set(context["supported_items"])
        exposure = dict(zip(report["item_order"], report["candidate_exposure"]))
        supported_order = iter(
            sorted(
                (item for item in target if item in support),
                key=lambda item: -exposure[item],
            )
        )
        exposure_sort = [
            next(supported_order) if item in support else item for item in target
        ]
        argmax = components[
            max(range(len(components)), key=lambda i: probabilities[i])
        ]["ranking"]
        shortcuts = {}
        for name, ranking in (
            ("argmax_component", argmax),
            ("supported_exposure_sort", exposure_sort),
        ):
            replacement = {
                **policy,
                "mixture": [{"probability": 1.0, "ranking": ranking}],
            }
            try:
                replacement_report = verify_policy(
                    replacement, expected_context=context
                )
                verdict = {"accepted": True, "regret": replacement_report["regret"]}
            except PolicyFormatError as exc:
                verdict = {"accepted": False, "reason": str(exc)}
            shortcuts[name] = {
                **verdict,
                "ndcg": metric(ranking),
                "difference_from_expected": metric(ranking) - expected,
            }
        reference = metric(target)
        rows[qid] = {
            "zero_idcg": ideal == 0,
            "reference_ndcg": reference,
            "expected_ndcg": expected,
            "expected_minus_reference": expected - reference,
            "exposure_evaluation_abs_error": abs(expected - by_exposure),
            "worst_case_expected_regret": report["regret"],
            "sample_mean_ndcg": math.fsum(counts[i] * metrics[i] for i in counts)
            / draws,
            "component_draw_counts": [counts[i] for i in range(len(components))],
            "shortcuts": shortcuts,
        }
    return {
        "schema": "metric-aligned-ranking.downstream-evaluation.v1",
        "query_count": len(rows),
        "zero_idcg_query_count": sum(row["zero_idcg"] for row in rows.values()),
        "macro_expected_ndcg": math.fsum(row["expected_ndcg"] for row in rows.values())
        / len(rows),
        "draws_per_query": draws,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "sampling_is_diagnostic_not_a_certificate": True,
        "queries": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--gains", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        inputs = {
            name: getattr(args, name).read_bytes()
            for name in ("requests", "policies", "gains")
        }
        parsed = {
            name: loads_policy_json(value.decode("utf-8-sig"))
            for name, value in inputs.items()
        }
        result = evaluate(
            parsed["policies"],
            parsed["requests"],
            parsed["gains"],
            draws=args.draws,
            seed=args.seed,
        )
        result["input_sha256"] = {
            name: hashlib.sha256(value).hexdigest() for name, value in inputs.items()
        }
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"downstream evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
