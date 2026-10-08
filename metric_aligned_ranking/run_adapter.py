"""TREC run-file adapter with explicit partial coverage and no qrels input."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .policy_io import (
    PolicyFormatError,
    PolicyOptimizationError,
    loads_policy_json,
    optimize_policy,
    validate_context,
    verify_policy,
)

BATCH_SCHEMA = "metric-aligned-ranking.batch.v1"


def read_trec_run(text: str, *, role: str = "run") -> dict:
    """Read qid Q0 docid rank score tag; preserve explicit rank, not score sort.

    Nonnegative unique ordinal ranks may start at zero or one and have gaps.
    Input line order is irrelevant. Duplicate query/document entries fail.
    """
    queries = {}
    ranks = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        location = f"{role}: line {line_number}"
        if len(fields) != 6 or fields[1] != "Q0":
            raise PolicyFormatError(f"{location}: expected six-column TREC run with Q0")
        qid, _, docid, raw_rank, raw_score, _ = fields
        location += f", query {qid!r}, document {docid!r}"
        try:
            rank = int(raw_rank)
        except ValueError as exc:
            raise PolicyFormatError(
                f"{location}: rank must be an integer; got {raw_rank!r}"
            ) from exc
        if rank < 0:
            raise PolicyFormatError(f"{location}: rank must be nonnegative; got {rank}")
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise PolicyFormatError(
                f"{location}: score must be numeric; got {raw_score!r}"
            ) from exc
        if not math.isfinite(score):
            raise PolicyFormatError(
                f"{location}: score must be finite; got {raw_score!r}"
            )
        documents = queries.setdefault(qid, {})
        used_ranks = ranks.setdefault(qid, set())
        if docid in documents:
            raise PolicyFormatError(f"{location}: duplicate document within query")
        if rank in used_ranks:
            raise PolicyFormatError(f"{location}: duplicate rank {rank} within query")
        documents[docid] = {"rank": rank, "score": score}
        used_ranks.add(rank)
    return queries


def prepare_runs(
    target_text: str, source_text: str, *, cutoff: int, risk_budget: float
) -> dict:
    """Build query requests; absent source rows remain unsupported, not score zero."""
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise PolicyFormatError("cutoff must be a positive integer")
    targets = read_trec_run(target_text, role="target run")
    sources = read_trec_run(source_text, role="source run")
    if not targets:
        raise PolicyFormatError("target run is empty")
    if not set(sources) <= set(targets):
        raise PolicyFormatError(
            f"source run: queries absent from target run: {sorted(set(sources) - set(targets))!r}"
        )
    requests = {}
    for qid in sorted(targets):
        target = targets[qid]
        source = sources.get(qid, {})
        if not set(source) <= set(target):
            raise PolicyFormatError(
                f"source run: query {qid!r}: documents outside target candidate set: "
                f"{sorted(set(source) - set(target))!r}"
            )
        ranking = sorted(target, key=lambda item: target[item]["rank"])
        support = [item for item in ranking if item in source]
        context = validate_context(
            {
                "target_ranking": ranking,
                "supported_items": support,
                "position_weights": [
                    1 / math.log2(i + 2) for i in range(min(cutoff, len(ranking)))
                ],
                "risk_budget": risk_budget,
            }
        )
        requests[qid] = {
            "context": context,
            "source_scores": {item: source[item]["score"] for item in support},
        }
    return requests


def _requests(requests: object) -> dict:
    if not isinstance(requests, dict) or not requests:
        raise PolicyFormatError("requests must be a nonempty query mapping")
    for qid, request in requests.items():
        if not isinstance(qid, str) or not qid:
            raise PolicyFormatError("query IDs must be nonempty strings")
        if not isinstance(request, dict) or set(request) != {
            "context",
            "source_scores",
        }:
            raise PolicyFormatError(
                "each request must contain context and source_scores"
            )
        context = validate_context(request["context"])
        scores = request["source_scores"]
        if not isinstance(scores, dict) or set(scores) != set(
            context["supported_items"]
        ):
            raise PolicyFormatError("source score keys must equal supported_items")
        for value in scores.values():
            try:
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                )
            except OverflowError:
                valid = False
            if not valid:
                raise PolicyFormatError("source scores must be finite numbers")
    return requests


def optimize_runs(requests: object, *, formulation: str = "compact") -> dict:
    requests = _requests(requests)
    policies = {}
    for qid, request in requests.items():
        try:
            policies[qid] = optimize_policy(
                request["context"],
                source_scores=request["source_scores"],
                formulation=formulation,
            )
        except PolicyFormatError as exc:
            raise PolicyFormatError(f"query {qid!r}: {exc}") from exc
        except PolicyOptimizationError as exc:
            raise PolicyOptimizationError(f"query {qid!r}: {exc}") from exc
    return {"schema": BATCH_SCHEMA, "policies": policies}


def verify_runs(batch: object, *, expected_requests: object) -> dict:
    requests = _requests(expected_requests)
    if (
        not isinstance(batch, dict)
        or set(batch) != {"schema", "policies"}
        or batch["schema"] != BATCH_SCHEMA
    ):
        raise PolicyFormatError("invalid batch schema")
    policies = batch["policies"]
    if not isinstance(policies, dict) or set(policies) != set(requests):
        raise PolicyFormatError("policy queries must exactly match expected queries")
    reports = {}
    for qid, request in requests.items():
        try:
            reports[qid] = verify_policy(
                policies[qid], expected_context=request["context"]
            )
        except PolicyFormatError as exc:
            raise PolicyFormatError(f"query {qid!r}: {exc}") from exc
    return {"valid": True, "query_count": len(reports), "reports": reports}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("target", type=Path)
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--cutoff", type=int, default=10)
    prepare.add_argument("--risk-budget", type=float, required=True)
    solve = commands.add_parser("optimize")
    solve.add_argument("requests", type=Path)
    solve.add_argument(
        "--formulation", choices=["compact", "cutting-plane"], default="compact"
    )
    verify = commands.add_parser("verify")
    verify.add_argument("policies", type=Path)
    verify.add_argument("requests", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_runs(
                args.target.read_text(encoding="utf-8-sig"),
                args.source.read_text(encoding="utf-8-sig"),
                cutoff=args.cutoff,
                risk_budget=args.risk_budget,
            )
        else:
            requests = loads_policy_json(args.requests.read_text(encoding="utf-8-sig"))
            if args.command == "optimize":
                output = optimize_runs(requests, formulation=args.formulation)
            else:
                output = verify_runs(
                    loads_policy_json(args.policies.read_text(encoding="utf-8-sig")),
                    expected_requests=requests,
                )
        print(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False))
    except (ValueError, PolicyOptimizationError, OSError, UnicodeError) as exc:
        print(f"run adapter error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
