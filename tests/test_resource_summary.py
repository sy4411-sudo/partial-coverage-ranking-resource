import copy
import json

import pytest

from scripts.analysis.summarize_resource_benchmark import (
    DEFAULT_RECORD,
    main,
    markdown,
    summarize,
)


def record():
    return json.loads(DEFAULT_RECORD.read_text(encoding="utf-8"))


def test_saved_evidence_reproduces_primary_counts_and_table():
    result = summarize(record())
    assert result["combinations"] == 26
    assert result["status_counts"] == {"ok": 26}
    assert result["completed_repetitions"] == 78
    assert result["paired_cases"] == 8
    assert result["max_reconstruction_abs_error"] == pytest.approx(
        3.4555691641457997e-15, abs=1e-28
    )
    table = markdown(result)
    assert "125.947" in table
    assert "33.354" in table
    assert "31.734" in table


def test_failed_workload_stays_visible_without_zero_latency():
    data = record()
    data["rows"][0] = {**data["rows"][0], "status": "timeout"}
    del data["rows"][0]["measurements"]
    result = summarize(data)
    assert result["status_counts"] == {"ok": 25, "timeout": 1}
    assert result["completed_repetitions"] == 75
    assert result["paired_cases"] == 7
    assert "complete_ms" not in result["rows"][0]
    assert "| empty | compact | timeout | 12 | 0 | n/a |" in markdown(result)


def test_pair_matching_requires_same_complete_specification():
    data = record()
    data["rows"][1]["case"]["seed"] += 1
    assert summarize(data)["paired_cases"] == 7


@pytest.mark.parametrize(
    "problem", ["duplicate", "incomplete", "nan", "negative", "status", "repeats"]
)
def test_malformed_or_incomplete_records_rejected(problem):
    data = record()
    if problem == "duplicate":
        data["rows"].append(copy.deepcopy(data["rows"][0]))
    elif problem == "incomplete":
        data["rows"][0]["measurements"].pop()
    elif problem in {"nan", "negative"}:
        data["rows"][0]["measurements"][0]["warm_end_to_end_seconds"] = (
            float("nan") if problem == "nan" else -1
        )
    elif problem == "status":
        data["rows"][0]["status"] = "silently_skipped"
    else:
        data["repeats"] = True
    with pytest.raises(ValueError):
        summarize(data)


def test_empty_result_does_not_claim_zero_error():
    data = record()
    for row in data["rows"]:
        row["status"] = "error"
        del row["measurements"]
    result = summarize(data)
    assert result["completed_repetitions"] == 0
    assert result["max_reconstruction_abs_error"] is None
    assert result["max_paired_median_objective_gap"] is None


def test_cli_reads_existing_record_only(capsys):
    assert main(["--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["combinations"] == 26
