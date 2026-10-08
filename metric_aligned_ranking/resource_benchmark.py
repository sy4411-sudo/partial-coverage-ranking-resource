"""Bounded, synthetic engineering measurements; no scientific labels or gates.

Each workload/formulation runs in a fresh child with a hard wall timeout.
Timing probes wrap existing functions in that child; bound source stays intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .policy_io import loads_policy_json, optimize_policy, verify_policy


def workloads(profile: str) -> list[dict]:
    if profile == "all":
        return workloads("scale") + workloads("matched")
    cases = [
        ("empty", 12, 0, 5, 0.02, "spread", "random"),
        ("single", 12, 1, 5, 0.02, "spread", "random"),
        ("below-cutoff", 20, 8, 5, 0.02, "tail", "random"),
        ("zero-budget", 12, 6, 5, 0.0, "spread", "random"),
        ("tied-scores", 12, 6, 5, 0.02, "spread", "tied"),
        ("small", 12, 6, 5, 0.02, "spread", "random"),
        ("small-full", 12, 12, 5, 0.1, "spread", "reverse"),
        ("medium", 40, 20, 10, 0.02, "spread", "random"),
        ("wide", 100, 50, 10, 0.02, "spread", "random"),
    ]
    if profile == "matched":
        cases = [
            ("n200-s100-a5", 200, 100, 10, 0.02, "anchored", "random"),
            ("n500-s100-a5", 500, 100, 10, 0.02, "anchored", "random"),
            ("n1000-s100-a5", 1000, 100, 10, 0.02, "anchored", "random"),
            ("n500-s500", 500, 500, 10, 0.02, "spread", "reverse"),
        ]
    elif profile == "scale":
        cases += [
            ("n200-s100", 200, 100, 10, 0.02, "spread", "random"),
            ("n500-s100", 500, 100, 10, 0.02, "spread", "random"),
            ("n1000-s100", 1000, 100, 10, 0.02, "spread", "random"),
            ("n200-s200", 200, 200, 10, 0.02, "spread", "reverse"),
            ("n500-s250", 500, 250, 10, 0.02, "spread", "random"),
        ]
    elif profile != "smoke":
        raise ValueError("unknown benchmark profile")
    keys = ("name", "n", "support", "cutoff", "budget", "placement", "scores")
    return [dict(zip(keys, values), seed=20261007) for values in cases]


def make_case(spec: dict) -> tuple[dict, dict]:
    n, count, cutoff = spec["n"], spec["support"], spec["cutoff"]
    if not 1 <= cutoff <= n or not 0 <= count <= n:
        raise ValueError("invalid workload dimensions")
    target = [f"synthetic-{i}" for i in range(n)]
    if spec["placement"] == "anchored":
        active = min(5, cutoff, count)
        tail = count - active
        if tail > n - cutoff:
            raise ValueError("not enough below-cutoff slots")
        indices = list(range(active)) + [
            cutoff + i * (n - cutoff) // tail for i in range(tail)
        ]
    elif spec["placement"] == "tail":
        if count > n - cutoff:
            raise ValueError("tail support overlaps cutoff")
        indices = list(range(n - count, n))
    else:
        indices = [i * n // count for i in range(count)]
    support = [target[i] for i in indices]
    rng = random.Random(spec["seed"])
    values = {
        item: (
            1.0
            if spec["scores"] == "tied"
            else (i / max(1, n - 1) if spec["scores"] == "reverse" else rng.random())
        )
        for i, item in zip(indices, support)
    }
    return {
        "target_ranking": target,
        "supported_items": support,
        "position_weights": [1 / math.log2(i + 2) for i in range(cutoff)],
        "risk_budget": spec["budget"],
    }, values


def peak_process_bytes() -> tuple[int | None, str]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return (
                counters.PeakWorkingSetSize,
                "Windows PeakWorkingSetSize; includes imports and warmup",
            )
        return None, "GetProcessMemoryInfo unavailable"
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(
            value * (1 if sys.platform == "darwin" else 1024)
        ), "ru_maxrss; includes imports and warmup"
    except (ImportError, OSError):
        return None, "peak memory unavailable"


@contextmanager
def probes(formulation: str):
    import scipy.optimize

    from . import partial_coverage_optimizer as core

    name = (
        "optimize_robust_slot_policy_compact"
        if formulation == "compact"
        else "optimize_robust_slot_policy"
    )
    original_lp = scipy.optimize.linprog
    original_recovery = core._birkhoff_decomposition
    original_solve = getattr(core, name)
    stats = {"lp_seconds": 0.0, "birkhoff_seconds": 0.0, "lp_calls": 0}

    def timed_lp(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_lp(*args, **kwargs)
        finally:
            stats["lp_seconds"] += time.perf_counter() - started
            stats["lp_calls"] += 1

    def timed_recovery(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_recovery(*args, **kwargs)
        finally:
            stats["birkhoff_seconds"] += time.perf_counter() - started

    def captured_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        stats["solver_exposure"] = result.candidate_exposure
        return result

    scipy.optimize.linprog = timed_lp
    core._birkhoff_decomposition = timed_recovery
    setattr(core, name, captured_solve)
    try:
        yield stats
    finally:
        scipy.optimize.linprog = original_lp
        core._birkhoff_decomposition = original_recovery
        setattr(core, name, original_solve)


def run_worker(spec: dict, formulation: str, repeats: int) -> dict:
    started = time.perf_counter()
    import numpy
    import scipy
    import scipy.optimize

    import_seconds = time.perf_counter() - started
    warmup, warmup_scores = make_case(workloads("smoke")[5])
    started = time.perf_counter()
    optimize_policy(warmup, source_scores=warmup_scores, formulation=formulation)
    warmup_seconds = time.perf_counter() - started
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        context, scores = make_case(spec)
        construction_seconds = time.perf_counter() - started
        with probes(formulation) as timing:
            started = time.perf_counter()
            policy = optimize_policy(
                context, source_scores=scores, formulation=formulation
            )
            policy_seconds = time.perf_counter() - started
        started = time.perf_counter()
        encoded = json.dumps(policy, ensure_ascii=True, allow_nan=False)
        received = loads_policy_json(encoded)
        serialization_seconds = time.perf_counter() - started
        started = time.perf_counter()
        report = verify_policy(received, expected_context=context)
        verification_seconds = time.perf_counter() - started
        exposure = report["candidate_exposure"]
        utility = math.fsum(
            scores.get(item, 0) * exposure[i]
            for i, item in enumerate(report["item_order"])
        )
        reconstruction_error = max(
            abs(x - y) for x, y in zip(exposure, timing.pop("solver_exposure"))
        )
        measurements.append(
            {
                **timing,
                "construction_seconds": construction_seconds,
                "policy_seconds": policy_seconds,
                "serialization_roundtrip_seconds": serialization_seconds,
                "verification_seconds": verification_seconds,
                "warm_end_to_end_seconds": construction_seconds
                + policy_seconds
                + serialization_seconds
                + verification_seconds,
                "serialized_bytes": len(encoded.encode("utf-8")),
                "mixture_size": report["mixture_size"],
                "regret": report["regret"],
                "source_utility": utility,
                "reconstruction_max_abs_error": reconstruction_error,
            }
        )
    peak, memory_scope = peak_process_bytes()
    return {
        "status": "ok",
        "active_supported_slots": len(
            set(context["supported_items"])
            & set(context["target_ranking"][: len(context["position_weights"])])
        ),
        "import_seconds": import_seconds,
        "warmup_seconds": warmup_seconds,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "peak_process_bytes": peak,
        "memory_scope": memory_scope,
        "measurements": measurements,
    }


def run_isolated(spec: dict, formulation: str, repeats: int, timeout: float) -> dict:
    command = [
        sys.executable,
        "-m",
        __spec__.name,
        "--worker",
        json.dumps(spec),
        "--formulation",
        formulation,
        "--repeats",
        str(repeats),
    ]
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        try:
            row = json.loads(result.stdout)
        except json.JSONDecodeError:
            row = {
                "status": "worker_error",
                "returncode": result.returncode,
                "error": result.stderr[-2000:],
            }
        if result.returncode != 0 and row.get("status") == "ok":
            row = {"status": "worker_error", "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        row = {"status": "timeout", "timeout_seconds": timeout}
    return {
        "case": spec,
        "formulation": formulation,
        "cold_process_seconds": time.perf_counter() - started,
        **row,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["smoke", "scale", "matched", "all"], default="smoke"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--formulation", choices=["compact", "cutting-plane"], default="compact"
    )
    args = parser.parse_args(argv)
    if (
        not 1 <= args.repeats <= 20
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("repeats must be 1..20 and timeout finite and positive")
    if args.worker:
        try:
            output = run_worker(json.loads(args.worker), args.formulation, args.repeats)
        except Exception as exc:  # noqa: BLE001 - Record all child failures; never call them feasible.
            output = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        print(json.dumps(output, allow_nan=False))
        return 0 if output["status"] == "ok" else 1
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path")
    rows = []
    for spec in workloads(args.profile):
        formulations = ["compact", "cutting-plane"] if spec["n"] <= 40 else ["compact"]
        for formulation in formulations:
            row = run_isolated(spec, formulation, args.repeats, args.timeout)
            rows.append(row)
            print(
                f"{spec['name']} / {formulation}: {row['status']}",
                file=sys.stderr,
                flush=True,
            )
    root = Path(__file__).resolve().parent
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "resource_benchmark.py",
            "policy_io.py",
            "partial_coverage_optimizer.py",
            "ndcg_risk.py",
        )
    }
    output = {
        "schema": "ranking.resource-engineering.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "synthetic engineering only; no held-out or natural-domain inference",
        "profile": args.profile,
        "repeats": args.repeats,
        "timeout_seconds_per_child": args.timeout,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_env_limit": 1,
        "source_sha256": hashes,
        "rows": rows,
    }
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
