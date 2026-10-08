"""Regenerate descriptive engineering tables from saved synthetic measurements.

This reader does not run an optimizer or open a dataset. Failed workloads stay
visible, and missing measurements are never converted into zero latency.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD = ROOT / "records/resource/engineering-resource-v2-20261007.json"
TIMINGS = {
    "complete_ms": "warm_end_to_end_seconds",
    "lp_ms": "lp_seconds",
    "birkhoff_ms": "birkhoff_seconds",
    "verification_ms": "verification_seconds",
}


def number(value, name, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(
            f"{name} must be finite" + (" and nonnegative" if nonnegative else "")
        )
    return value


def summarize(data):
    if data["schema"] != "ranking.resource-engineering.v1":
        raise ValueError("unsupported engineering record schema")
    repeats = data["repeats"]
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(data["rows"], list) or not data["rows"]:
        raise ValueError("rows must be a nonempty list")
    rows, seen, matched, residuals = [], set(), {}, []
    counts = Counter()
    for raw in data["rows"]:
        spec, method, status = raw["case"], raw["formulation"], raw["status"]
        key = (spec["name"], method)
        if key in seen:
            raise ValueError(f"duplicate workload/formulation: {key}")
        seen.add(key)
        if status not in {"ok", "timeout", "error", "worker_error"}:
            raise ValueError(f"unknown worker status: {status}")
        counts[status] += 1
        row = {
            "case": spec["name"],
            "formulation": method,
            "status": status,
            "n": spec["n"],
            "support": spec["support"],
        }
        if status == "ok":
            measures = raw["measurements"]
            if not isinstance(measures, list) or len(measures) != repeats:
                raise ValueError(f"incomplete successful measurements: {key}")
            for label, field in TIMINGS.items():
                row[label] = 1000 * statistics.median(
                    number(m[field], field) for m in measures
                )
            row["active_supported_slots"] = raw["active_supported_slots"]
            peak = raw["peak_process_bytes"]
            row["peak_mib"] = (
                None if peak is None else number(peak, "peak memory") / 2**20
            )
            row["median_serialized_bytes"] = statistics.median(
                number(m["serialized_bytes"], "serialized bytes") for m in measures
            )
            row["median_mixture_size"] = statistics.median(
                number(m["mixture_size"], "mixture size") for m in measures
            )
            residuals.extend(
                number(m["reconstruction_max_abs_error"], "residual") for m in measures
            )
            # Match full workload definitions, not just display names.
            fingerprint = json.dumps(spec, sort_keys=True, allow_nan=False)
            matched.setdefault(fingerprint, {})[method] = statistics.median(
                number(m["source_utility"], "source utility", nonnegative=False)
                for m in measures
            )
        rows.append(row)
    gaps = [
        abs(values["compact"] - values["cutting-plane"])
        for values in matched.values()
        if {"compact", "cutting-plane"} <= values.keys()
    ]
    return {
        "scope": "descriptive synthetic engineering; no new experiment",
        "combinations": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "completed_repetitions": counts["ok"] * repeats,
        "paired_cases": len(gaps),
        "max_paired_median_objective_gap": max(gaps, default=None),
        "max_reconstruction_abs_error": max(residuals, default=None),
        "rows": rows,
    }


def markdown(result):
    lines = [
        "# Synthetic engineering measurements",
        "",
        (
            f"Workload/formulation combinations: {result['combinations']}. "
            f"Statuses: {json.dumps(result['status_counts'], sort_keys=True)}. "
            f"Completed repetitions: {result['completed_repetitions']}."
        ),
        "",
        "Times are medians in milliseconds; peak process memory includes imports and warmup.",
        "",
        "| Case | Method | Status | n | Supported | Active | Complete ms | LP ms | Birkhoff ms | Verify ms | Peak MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        values = [
            str(row[k]).replace("|", "\\|").replace("\n", " ")
            for k in ("case", "formulation", "status", "n", "support")
        ]
        values.append(str(row.get("active_supported_slots", "n/a")))
        values.extend(f"{row[k]:.3f}" if k in row else "n/a" for k in TIMINGS)
        peak = row.get("peak_mib")
        values.append("n/a" if peak is None else f"{peak:.2f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            f"Paired workloads: {result['paired_cases']}.",
            f"Maximum paired median source-objective gap: {result['max_paired_median_objective_gap']}.",
            f"Maximum reconstruction error: {result['max_reconstruction_abs_error']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args(argv)
    try:
        result = summarize(json.loads(args.record.read_text(encoding="utf-8-sig")))
        print(
            markdown(result)
            if args.format == "markdown"
            else json.dumps(result, indent=2, allow_nan=False)
        )
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f"benchmark summary error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
