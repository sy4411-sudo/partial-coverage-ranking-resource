import importlib.util
import json
from pathlib import Path

import pytest

from metric_aligned_ranking.policy_io import loads_policy_json
from metric_aligned_ranking.run_adapter import verify_runs

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "retrieval_workflow", ROOT / "examples/retrieval_workflow.py"
)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


@pytest.mark.parametrize("support", [0, 1, 6, 12])
def test_end_to_end(tmp_path, support):
    pytest.importorskip("rank_bm25")
    pytest.importorskip("sklearn")
    data = ROOT / "examples/retrieval"
    out = tmp_path / "result"
    receipt = workflow.run(
        data / "corpus.json", data / "queries.json", out, source_limit=support
    )
    assert receipt["valid"] and receipt["labels_used"] is False
    assert len(receipt["counts"]) == 4
    assert all(
        row["second_stage_scores"] == support for row in receipt["counts"].values()
    )
    expected = loads_policy_json((out / "requests.json").read_text())
    policies = loads_policy_json((out / "policies.json").read_text())
    assert verify_runs(policies, expected_requests=expected)["valid"]
    with pytest.raises(ValueError, match="already exists"):
        workflow.run(data / "corpus.json", data / "queries.json", out)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": "a b", "text": "x"}],
        [{"id": "a", "text": ""}],
        [{"id": "a", "text": "x", "label": 1}],
        [{"id": "a", "text": "x"}] * 2,
    ],
)
def test_invalid_text_inputs(tmp_path, rows):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        workflow.read_texts(path)
