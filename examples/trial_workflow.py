"""Run invented or user-supplied TREC inputs through the installed resource."""

import argparse
import json
from pathlib import Path

from metric_aligned_ranking.ndcg_risk import MetricAlignedRankingError
from metric_aligned_ranking.policy_io import PolicyOptimizationError
from metric_aligned_ranking.run_adapter import optimize_runs, prepare_runs, verify_runs


def main(argv=None):
    root = Path(__file__).resolve().parent
    # In the trial bundle this script is at its root; in the repo it is in examples.
    examples = root / "examples" if (root / "examples").is_dir() else root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=examples / "target.trec")
    parser.add_argument("--source", type=Path, default=examples / "source_partial.trec")
    parser.add_argument("--output", type=Path, default=Path("trial-output"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists; choose a new --output")
    try:
        requests = prepare_runs(
            args.target.read_text(encoding="utf-8-sig"),
            args.source.read_text(encoding="utf-8-sig"),
            cutoff=2,
            risk_budget=0.1,
        )
        policies = optimize_runs(requests)
        report = verify_runs(policies, expected_requests=requests)
        args.output.mkdir(parents=True)
        for name, value in (
            ("requests", requests),
            ("policies", policies),
            ("report", report),
        ):
            (args.output / f"{name}.json").write_text(
                json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    except (
        MetricAlignedRankingError,
        PolicyOptimizationError,
        OSError,
        UnicodeError,
    ) as exc:
        parser.exit(2, f"trial error: {exc}\n")
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "query_count": report["query_count"],
                "coverage": {
                    qid: {
                        "supported": len(row["supported_items"]),
                        "unsupported": len(row["unsupported_items"]),
                    }
                    for qid, row in report["reports"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
