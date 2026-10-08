import json
import math
import subprocess

import pytest

from metric_aligned_ranking import resource_benchmark as bench
from metric_aligned_ranking.policy_io import optimize_policy, verify_policy


def test_workload_manifest_is_deterministic_and_bounded():
    assert bench.workloads("smoke") == bench.workloads("smoke")
    assert len(bench.workloads("smoke")) == 9
    assert len(bench.workloads("scale")) == 14
    assert len(bench.workloads("all")) == 18
    for spec in bench.workloads("scale"):
        context, scores = bench.make_case(spec)
        assert len(context["target_ranking"]) == spec["n"]
        assert len(scores) == spec["support"]
        assert set(scores) == set(context["supported_items"])
        assert set(context) == {
            "target_ranking",
            "supported_items",
            "position_weights",
            "risk_budget",
        }


def test_tail_support_has_no_exposure():
    context, _ = bench.make_case(bench.workloads("smoke")[2])
    assert not set(context["supported_items"]) & set(context["target_ranking"][:5])


def test_matched_cases_hold_active_support_and_scores_fixed():
    cases = [bench.make_case(spec) for spec in bench.workloads("matched")[:3]]
    for context, scores in cases:
        assert len(scores) == 100
        assert (
            len(set(context["supported_items"]) & set(context["target_ranking"][:10]))
            == 5
        )
    assert (
        list(cases[0][1].values())
        == list(cases[1][1].values())
        == list(cases[2][1].values())
    )


def test_timeout_is_recorded_not_called_infeasible(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(bench.subprocess, "run", timeout)
    result = bench.run_isolated(bench.workloads("smoke")[0], "compact", 1, 0.01)
    assert result["status"] == "timeout"
    assert result["timeout_seconds"] == 0.01
    assert "measurements" not in result


def test_crash_and_reported_error_are_retained(monkeypatch):
    for stdout, expected in [
        ("", "worker_error"),
        (json.dumps({"status": "error", "error": "numeric"}), "error"),
    ]:
        monkeypatch.setattr(
            bench.subprocess,
            "run",
            lambda *args, stdout=stdout, **kwargs: subprocess.CompletedProcess(
                args[0], 1, stdout=stdout, stderr="crash"
            ),
        )
        assert (
            bench.run_isolated(bench.workloads("smoke")[0], "compact", 1, 1)["status"]
            == expected
        )


def test_probes_restore_original_functions():
    scipy = pytest.importorskip("scipy.optimize")
    from metric_aligned_ranking import partial_coverage_optimizer as core

    originals = (
        scipy.linprog,
        core._birkhoff_decomposition,
        core.optimize_robust_slot_policy_compact,
    )
    with pytest.raises(RuntimeError), bench.probes("compact"):
        assert scipy.linprog is not originals[0]
        raise RuntimeError("deliberate test failure")
    assert originals == (
        scipy.linprog,
        core._birkhoff_decomposition,
        core.optimize_robust_slot_policy_compact,
    )


def test_worker_records_real_end_to_end_measurements():
    pytest.importorskip("scipy")
    result = bench.run_worker(bench.workloads("smoke")[5], "compact", 1)
    row = result["measurements"][0]
    assert result["status"] == "ok"
    assert row["lp_calls"] > 0
    assert row["warm_end_to_end_seconds"] >= row["policy_seconds"] >= row["lp_seconds"]
    assert row["reconstruction_max_abs_error"] < 1e-8
    assert row["mixture_size"] > 1
    assert row["serialized_bytes"] > 0
    assert result["peak_process_bytes"] is None or result["peak_process_bytes"] > 0


@pytest.mark.parametrize(
    "spec", bench.workloads("smoke")[:8], ids=lambda spec: spec["name"]
)
def test_formulations_match_objective_not_mixture_bytes(spec):
    pytest.importorskip("scipy")
    context, scores = bench.make_case(spec)
    utilities = []
    for method in ("compact", "cutting-plane"):
        result = verify_policy(
            optimize_policy(context, source_scores=scores, formulation=method),
            expected_context=context,
        )
        utilities.append(
            math.fsum(
                scores.get(item, 0) * e
                for item, e in zip(result["item_order"], result["candidate_exposure"])
            )
        )
    assert utilities[0] == pytest.approx(utilities[1], abs=1e-8)


def test_output_refuses_overwrite(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("untouched")
    with pytest.raises(SystemExit):
        bench.main(["--output", str(output)])
    assert output.read_text() == "untouched"
