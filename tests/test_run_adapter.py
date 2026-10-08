import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from metric_aligned_ranking.policy_io import PolicyFormatError
from metric_aligned_ranking.run_adapter import (
    main,
    optimize_runs,
    prepare_runs,
    read_trec_run,
    verify_runs,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = (ROOT / "examples/target.trec").read_text()
SOURCE = (ROOT / "examples/source_partial.trec").read_text()


def test_coverage_not_zero_imputation_and_missing_query_supported():
    requests = prepare_runs(TARGET, SOURCE, cutoff=10, risk_budget=0.1)
    assert requests["toy-1"]["context"]["supported_items"] == ["doc-a", "doc-c"]
    assert requests["toy-2"]["source_scores"] == {}
    assert requests["toy-2"]["context"]["supported_items"] == []
    assert len(requests["toy-2"]["context"]["position_weights"]) == 2


def test_explicit_rank_not_line_order_or_score_sort():
    text = "q Q0 z 2 100 run\nq Q0 x 0 -10 run\nq Q0 y 1 0 run\n"
    request = prepare_runs(text, "", cutoff=2, risk_budget=0)["q"]
    assert request["context"]["target_ranking"] == ["x", "y", "z"]


@pytest.mark.parametrize(
    "text",
    [
        "q 0 d 1",
        "q Q0 d 0 nan run",
        "q Q0 d -1 1 run",
        "q Q0 d 0 1 run\nq Q0 d 1 2 run",
        "q Q0 d 0 1 run\nq Q0 e 0 2 run",
        "q Q0 d 0.5 1 run",
    ],
)
def test_malformed_runs_and_qrels_are_rejected(text):
    with pytest.raises(PolicyFormatError):
        read_trec_run(text)


@pytest.mark.parametrize(
    "source", ["other Q0 doc-a 1 0.1 run", "toy-1 Q0 absent 1 0.1 run"]
)
def test_source_outside_candidate_universe_rejected(source):
    with pytest.raises(PolicyFormatError):
        prepare_runs(TARGET, source, cutoff=2, risk_budget=0.1)


def test_empty_target_and_bad_cutoff():
    with pytest.raises(PolicyFormatError):
        prepare_runs("", "", cutoff=2, risk_budget=0.1)
    with pytest.raises(PolicyFormatError):
        prepare_runs(TARGET, SOURCE, cutoff=True, risk_budget=0.1)


def test_roundtrip_and_dropped_queries_rejected():
    pytest.importorskip("scipy")
    requests = prepare_runs(TARGET, SOURCE, cutoff=2, risk_budget=0.1)
    original = copy.deepcopy(requests)
    batch = optimize_runs(requests)
    result = verify_runs(batch, expected_requests=requests)
    assert result["query_count"] == 2
    assert result["reports"]["toy-2"]["regret"] == 0
    assert requests == original
    del batch["policies"]["toy-2"]
    with pytest.raises(PolicyFormatError, match="exactly"):
        verify_runs(batch, expected_requests=requests)


def test_zero_source_score_is_covered():
    request = prepare_runs(TARGET, "toy-1 Q0 doc-a 1 0 run", cutoff=2, risk_budget=0.1)[
        "toy-1"
    ]
    assert request["context"]["supported_items"] == ["doc-a"]
    assert request["source_scores"] == {"doc-a": 0.0}


def test_batch_cli_prepare_optimize_then_solver_free_verify(tmp_path):
    pytest.importorskip("scipy")
    command = [sys.executable, "-m", "metric_aligned_ranking.run_adapter"]
    prepared = subprocess.run(
        command
        + [
            "prepare",
            "examples/target.trec",
            "examples/source_partial.trec",
            "--cutoff",
            "2",
            "--risk-budget",
            "0.1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    requests = tmp_path / "requests.json"
    requests.write_text(prepared.stdout, encoding="utf-8")
    solved = subprocess.run(
        command + ["optimize", str(requests)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    policies = tmp_path / "policies.json"
    policies.write_text(solved.stdout, encoding="utf-8")
    verified = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "metric_aligned_ranking.run_adapter",
            "verify",
            str(policies),
            str(requests),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(verified.stdout)["query_count"] == 2


def test_cli_error_is_nonzero(tmp_path, capsys):
    assert main(["optimize", str(tmp_path / "absent")]) == 2
    assert "run adapter error" in capsys.readouterr().err


@pytest.mark.parametrize("role", ["target", "source"])
@pytest.mark.parametrize(
    "bad_row, diagnostic",
    [
        ("q Q0 a 2 1 run", "duplicate document"),
        ("q Q0 b 1 1 run", "duplicate rank 1"),
        ("q Q0 b -1 1 run", "rank must be nonnegative"),
        ("q Q0 b 1.5 1 run", "rank must be an integer"),
        ("q Q0 b 2 nan run", "score must be finite"),
        ("q Q0 b 2 invalid run", "score must be numeric"),
    ],
)
def test_run_errors_identify_role_line_query_and_field(role, bad_row, diagnostic):
    good = "q Q0 a 1 1 run\nq Q0 b 2 0 run\n"
    bad = "q Q0 a 1 1 run\n" + bad_row + "\n"
    with pytest.raises(PolicyFormatError) as error:
        prepare_runs(
            bad if role == "target" else good,
            bad if role == "source" else "",
            cutoff=2,
            risk_budget=0.1,
        )
    message = str(error.value)
    assert f"{role} run: line 2" in message
    assert "query 'q'" in message
    assert diagnostic in message


@pytest.mark.parametrize(
    "probabilities, required",
    [
        ((0.8, 0.1), ["actual sum=", "absolute error="]),
        ((0.1, 0.9), ["regret=", "budget=", "tolerance="]),
    ],
)
def test_batch_error_identifies_query_and_numeric_values(probabilities, required):
    requests = prepare_runs(TARGET, SOURCE, cutoff=2, risk_budget=0.1)
    policy = json.loads((ROOT / "examples/policy_mixture.json").read_text())
    for component, probability in zip(policy["mixture"], probabilities):
        component["probability"] = probability
    from metric_aligned_ranking.run_adapter import BATCH_SCHEMA

    with pytest.raises(PolicyFormatError) as error:
        verify_runs(
            {"schema": BATCH_SCHEMA, "policies": {"toy-1": policy}},
            expected_requests={"toy-1": requests["toy-1"]},
        )
    assert "query 'toy-1'" in str(error.value)
    for text in required:
        assert text in str(error.value)


def test_reports_expose_support_without_imputing_missing_scores():
    request = prepare_runs(TARGET, SOURCE, cutoff=2, risk_budget=0.1)["toy-1"]
    from metric_aligned_ranking.policy_io import verify_policy

    policy = json.loads((ROOT / "examples/policy_mixture.json").read_text())
    report = verify_policy(policy, expected_context=request["context"])
    assert report["supported_items"] == ["doc-a", "doc-c"]
    assert report["unsupported_items"] == ["fixed-b", "fixed-d"]
