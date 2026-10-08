import copy
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from metric_aligned_ranking.policy_io import (
    SCHEMA,
    PolicyFormatError,
    loads_policy_json,
    main,
    optimize_policy,
    validate_context,
    verify_policy,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context():
    return {
        "target_ranking": ["doc-a", "fixed-b", "doc-c", "fixed-d"],
        "supported_items": ["doc-a", "doc-c"],
        "position_weights": [1.0, 1.0 / math.log2(3)],
        "risk_budget": 0.1,
    }


@pytest.fixture
def policy(context):
    target = context["target_ranking"]
    return {
        "schema": SCHEMA,
        "context": copy.deepcopy(context),
        "mixture": [
            {"probability": 0.9, "ranking": list(target)},
            {
                "probability": 0.1,
                "ranking": [target[2], target[1], target[0], target[3]],
            },
        ],
    }


def test_round_trip_and_exhaustive_oracle(context, policy):
    report = verify_policy(
        loads_policy_json(json.dumps(policy)), expected_context=context
    )
    target, candidate = report["target_exposure"], report["candidate_exposure"]
    weights = context["position_weights"]
    losses = []
    for count in range(1, 5):
        for relevant in itertools.combinations(range(4), count):
            losses.append(
                sum(target[i] - candidate[i] for i in relevant)
                / sum(weights[: min(count, len(weights))])
            )
    assert report["regret"] == pytest.approx(max(losses))
    assert report["regret"] == pytest.approx(0.1)
    assert report["expected_policy_only"]
    assert not report["optimality_checked"]
    assert not report["empirical_utility_checked"]


@pytest.mark.parametrize(
    "field", ["risk_budget", "target_ranking", "supported_items", "position_weights"]
)
def test_context_cannot_authorize_itself(context, policy, field):
    replacements = {
        "risk_budget": 1.0,
        "target_ranking": list(reversed(context["target_ranking"])),
        "supported_items": list(context["target_ranking"]),
        "position_weights": [1.0],
    }
    policy["context"][field] = replacements[field]
    with pytest.raises(PolicyFormatError, match="differs"):
        verify_policy(policy, expected_context=context)


@pytest.mark.parametrize(
    "probability", [True, "0.1", -0.1, float("nan"), float("inf"), 1.01]
)
def test_bad_probabilities(context, policy, probability):
    policy["mixture"][1]["probability"] = probability
    with pytest.raises(PolicyFormatError):
        verify_policy(policy, expected_context=context)


def test_no_silent_repair_of_mass(context, policy):
    policy["mixture"][0]["probability"] = 0.8
    with pytest.raises(PolicyFormatError, match="sum to one"):
        verify_policy(policy, expected_context=context)


def test_small_mass_residual_is_explicit(context, policy):
    policy["mixture"][0]["probability"] += 2e-13
    report = verify_policy(policy, expected_context=context)
    assert report["probability_sum_before_normalization"] > 1
    assert sum(report["candidate_exposure"]) == pytest.approx(
        sum(context["position_weights"])
    )


def test_certificate_comes_from_actual_mixture(context, policy):
    policy["mixture"][0]["probability"] = 0.8
    policy["mixture"][1]["probability"] = 0.2
    with pytest.raises(PolicyFormatError, match="exceeds"):
        verify_policy(policy, expected_context=context)


def test_a_draw_is_not_the_certified_policy(context, policy):
    # This draw loses 1.0 if only doc-a is relevant, despite the 0.1 mixture budget.
    policy["mixture"] = [dict(policy["mixture"][1], probability=1.0)]
    with pytest.raises(PolicyFormatError, match="exceeds"):
        verify_policy(policy, expected_context=context)


def test_fixed_slot_violation_even_in_zero_mass_entry(context, policy):
    policy["mixture"].append(
        {"probability": 0, "ranking": list(reversed(context["target_ranking"]))}
    )
    with pytest.raises(PolicyFormatError, match="unsupported"):
        verify_policy(policy, expected_context=context)


@pytest.mark.parametrize(
    "ranking", [["doc-a"] * 4, ["doc-a", "fixed-b", "unknown", "fixed-d"], [0, 1, 2, 3]]
)
def test_invalid_component_ids(context, policy, ranking):
    policy["mixture"][0]["ranking"] = ranking
    with pytest.raises(PolicyFormatError):
        verify_policy(policy, expected_context=context)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "{"])
def test_strict_json(text):
    with pytest.raises(PolicyFormatError):
        loads_policy_json(text)


@pytest.mark.parametrize(
    "change",
    [
        {"risk_budget": True},
        {"risk_budget": -1},
        {"position_weights": [0]},
        {"position_weights": [0.5, 1]},
        {"position_weights": [1e308, 1e308]},
        {"supported_items": ["missing"]},
        {"target_ranking": []},
        {"outcome": 1},
    ],
)
def test_context_validation(context, change):
    context.update(change)
    with pytest.raises(PolicyFormatError):
        validate_context(context)


def test_claimed_certificate_is_not_an_input(context, policy):
    policy["regret"] = 0
    with pytest.raises(PolicyFormatError):
        verify_policy(policy, expected_context=context)


def test_verifier_without_site_packages(tmp_path, context, policy):
    context_path, policy_path = tmp_path / "context.json", tmp_path / "policy.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    command = [
        sys.executable,
        "-S",
        "-m",
        "metric_aligned_ranking.policy_io",
        "verify",
        str(policy_path),
        str(context_path),
    ]
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["valid"]
    policy["mixture"][1]["probability"] = 0.5
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert process.returncode == 2
    assert not process.stdout


def test_cli_missing_input(tmp_path, capsys):
    assert main(["verify", str(tmp_path / "missing"), str(tmp_path / "missing")]) == 2
    assert "policy error" in capsys.readouterr().err


def test_optimizer_roundtrip_and_no_input_mutation(context):
    pytest.importorskip("scipy")
    original = copy.deepcopy(context)
    policy = optimize_policy(context, source_scores={"doc-a": 0.1, "doc-c": 1.0})
    report = verify_policy(policy, expected_context=context)
    assert context == original
    assert report["regret"] == pytest.approx(0.1)
    assert report["mixture_size"] >= 2
    assert "source_scores" not in json.dumps(policy)


@pytest.mark.parametrize(
    "scores", [{}, {"doc-a": 1, "doc-c": True}, {"doc-a": 1, "doc-c": 2, "fixed-b": 0}]
)
def test_optimizer_rejects_bad_scores(context, scores):
    with pytest.raises(PolicyFormatError):
        optimize_policy(context, source_scores=scores)


@pytest.mark.parametrize("support", [[], ["doc-a"], ["fixed-d"]])
def test_degenerate_support(context, support):
    pytest.importorskip("scipy")
    context["supported_items"] = support
    policy = optimize_policy(context, source_scores=dict.fromkeys(support, 1.0))
    assert verify_policy(policy, expected_context=context)["regret"] == 0


def test_unicode_external_ids(context):
    pytest.importorskip("scipy")
    context["target_ranking"][0] = "\u6587\u6863"
    context["supported_items"][0] = "\u6587\u6863"
    policy = optimize_policy(context, source_scores={"\u6587\u6863": 0.1, "doc-c": 1.0})
    assert verify_policy(
        loads_policy_json(json.dumps(policy)), expected_context=context
    )["valid"]


def test_unknown_solver_fails_before_execution(context):
    with pytest.raises(PolicyFormatError, match="formulation"):
        optimize_policy(
            context, source_scores={"doc-a": 0.1, "doc-c": 1}, formulation="unknown"
        )


def test_solver_failure_has_public_error_type(context, monkeypatch):
    from metric_aligned_ranking import partial_coverage_optimizer as core
    from metric_aligned_ranking.policy_io import PolicyOptimizationError

    def fail(*args, **kwargs):
        raise core.RobustSlotRankingError("numerical failure")

    monkeypatch.setattr(core, "optimize_robust_slot_policy_compact", fail)
    with pytest.raises(PolicyOptimizationError, match="numerical failure"):
        optimize_policy(context, source_scores={"doc-a": 0.1, "doc-c": 1})
