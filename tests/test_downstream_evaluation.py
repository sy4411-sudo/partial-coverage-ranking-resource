import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from metric_aligned_ranking.policy_io import SCHEMA, loads_policy_json
from metric_aligned_ranking.run_adapter import BATCH_SCHEMA, optimize_runs

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"examples/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


consumer = load_example("downstream_evaluation")


def fixtures():
    folder = ROOT / "examples/evaluation"
    requests = loads_policy_json((folder / "requests.json").read_text())
    gains = loads_policy_json((folder / "gains.json").read_text())
    policies = {}
    for qid, request in requests.items():
        context = request["context"]
        p = context["risk_budget"] if context["supported_items"] else 0.0
        policies[qid] = {
            "schema": SCHEMA,
            "context": copy.deepcopy(context),
            "mixture": [
                {"probability": 1 - p, "ranking": ["a", "fixed", "b"]},
                {"probability": p, "ranking": ["b", "fixed", "a"]},
            ]
            if p
            else [{"probability": 1.0, "ranking": ["a", "fixed", "b"]}],
        }
    return {"schema": BATCH_SCHEMA, "policies": policies}, requests, gains


def test_hand_computable_endpoints_and_shortcuts():
    batch, requests, gains = fixtures()
    before = copy.deepcopy((batch, requests, gains))
    result = consumer.evaluate(batch, requests, gains, draws=1000, seed=7)
    assert (batch, requests, gains) == before
    assert result["macro_expected_ndcg"] == pytest.approx(0.44)
    assert result["zero_idcg_query_count"] == 1
    rows = result["queries"]
    for qid, value in {
        "reference_gain": 0.4,
        "source_gain": 0.6,
        "small_budget": 0.2,
        "no_coverage": 1,
        "zero_gain": 0,
    }.items():
        assert rows[qid]["expected_ndcg"] == pytest.approx(value)
        assert sum(rows[qid]["component_draw_counts"]) == 1000
        assert rows[qid]["exposure_evaluation_abs_error"] < 1e-12
    assert rows["reference_gain"]["expected_minus_reference"] == pytest.approx(-0.6)
    for shortcut in ("argmax_component", "supported_exposure_sort"):
        assert rows["reference_gain"]["shortcuts"][shortcut]["accepted"] is False
        assert (
            "exceeds expected regret budget"
            in rows["reference_gain"]["shortcuts"][shortcut]["reason"]
        )
        assert rows["small_budget"]["shortcuts"][shortcut]["accepted"] is True
        assert rows["small_budget"]["shortcuts"][shortcut][
            "difference_from_expected"
        ] == pytest.approx(-0.2)
    assert consumer.evaluate(batch, requests, gains, draws=1000, seed=7) == result


def test_normalized_mass_and_large_gains():
    batch, requests, gains = fixtures()
    for entry in batch["policies"]["reference_gain"]["mixture"]:
        entry["probability"] *= 1 - 5e-13
    gains["reference_gain"]["a"] = 1e308
    result = consumer.evaluate(batch, requests, gains, draws=10)
    assert result["queries"]["reference_gain"]["expected_ndcg"] == pytest.approx(0.4)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1", 10**1000])
def test_reject_invalid_gains(value):
    batch, requests, gains = fixtures()
    gains["reference_gain"]["a"] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        consumer.evaluate(batch, requests, gains)


@pytest.mark.parametrize(
    "change", ["missing_query", "extra_query", "missing_item", "extra_item"]
)
def test_complete_judgments_required(change):
    batch, requests, gains = fixtures()
    if change == "missing_query":
        gains.pop("reference_gain")
    elif change == "extra_query":
        gains["extra"] = {}
    elif change == "missing_item":
        gains["reference_gain"].pop("fixed")
    else:
        gains["reference_gain"]["extra"] = 0
    with pytest.raises(ValueError, match="exactly"):
        consumer.evaluate(batch, requests, gains)


def test_recipient_context_and_invalid_policy_checked_before_gains():
    batch, requests, _ = fixtures()
    batch["policies"]["reference_gain"]["context"]["risk_budget"] = 1
    with pytest.raises(ValueError, match="differs from expected"):
        consumer.evaluate(batch, requests, None)


@pytest.mark.parametrize(
    "draws,seed", [(0, 0), (-1, 0), (True, 0), (1.5, 0), (10, True), (10, "0")]
)
def test_invalid_sampling_parameters(draws, seed):
    with pytest.raises(ValueError):
        consumer.evaluate(*fixtures(), draws=draws, seed=seed)


def test_direct_core_and_public_interface_complete_same_task():
    pytest.importorskip("scipy")
    _, requests, gains = fixtures()
    direct = load_example("direct_solver_comparison")
    batches = [
        optimize_runs(requests),
        {
            "schema": BATCH_SCHEMA,
            "policies": {
                qid: direct.assemble_direct(req["context"], req["source_scores"])
                for qid, req in requests.items()
            },
        },
    ]
    for batch in batches:
        result = consumer.evaluate(batch, requests, gains, draws=100)
        assert result["macro_expected_ndcg"] == pytest.approx(0.44)


def test_text_producer_connects_to_evaluation(tmp_path):
    pytest.importorskip("rank_bm25")
    pytest.importorskip("sklearn")
    producer = load_example("retrieval_workflow")
    data = ROOT / "examples/retrieval"
    output = tmp_path / "retrieval"
    producer.run(data / "corpus.json", data / "queries.json", output)
    result = consumer.evaluate(
        json.loads((output / "policies.json").read_text()),
        json.loads((output / "requests.json").read_text()),
        json.loads((data / "illustrative_gains.json").read_text()),
        draws=100,
    )
    assert result["query_count"] == 4
    assert all(
        row["exposure_evaluation_abs_error"] < 1e-12
        for row in result["queries"].values()
    )


def test_cli_no_solver_dependency_nonoverwrite_and_strict_json(tmp_path):
    batch, requests, gains = fixtures()
    for name, data in (("policies", batch), ("requests", requests), ("gains", gains)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    # -S removes site packages. Only the checkout package and stdlib are available.
    script = "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    args = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        script,
        str(ROOT),
        str(ROOT / "examples/downstream_evaluation.py"),
    ]
    for name in ("policies", "requests", "gains"):
        args += [f"--{name}", str(tmp_path / f"{name}.json")]
    output = tmp_path / "report.json"
    args += ["--output", str(output), "--draws", "100"]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert set(report["input_sha256"]) == {"requests", "policies", "gains"}
    before = output.read_bytes()
    assert subprocess.run(args, capture_output=True, check=False).returncode == 2
    assert output.read_bytes() == before
    (tmp_path / "gains.json").write_text('{"duplicate": 0, "duplicate": 1}')
    args[-3] = str(tmp_path / "other.json")
    failed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert failed.returncode == 2 and "duplicate JSON key" in failed.stderr
    assert not (tmp_path / "other.json").exists()
