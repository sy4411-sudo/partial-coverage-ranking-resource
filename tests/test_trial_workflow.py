import json

import pytest

from examples import trial_workflow as trial
from metric_aligned_ranking.policy_io import PolicyOptimizationError


def inputs(tmp_path):
    target = tmp_path / "target.trec"
    source = tmp_path / "source.trec"
    target.write_text("q Q0 a 1 1 toy\nq Q0 b 2 0 toy\n", encoding="utf-8")
    source.write_text("", encoding="utf-8")
    output = tmp_path / "output"
    return target, source, output


def arguments(target, source, output):
    return ["--target", str(target), "--source", str(source), "--output", str(output)]


def test_trial_completes_without_source_scores(tmp_path, capsys):
    target, source, output = inputs(tmp_path)
    trial.main(arguments(target, source, output))
    assert json.loads(capsys.readouterr().out) == {
        "valid": True,
        "query_count": 1,
        "coverage": {"q": {"supported": 0, "unsupported": 2}},
    }
    report = json.loads((output / "report.json").read_text())
    assert report["reports"]["q"]["regret"] == 0
    assert {p.name for p in output.iterdir()} == {
        "requests.json",
        "policies.json",
        "report.json",
    }


@pytest.mark.parametrize(
    "problem", ["duplicate_rank", "missing_file", "encoding", "solver"]
)
def test_expected_failures_are_concise(tmp_path, capsys, monkeypatch, problem):
    target, source, output = inputs(tmp_path)
    if problem == "duplicate_rank":
        target.write_text("q Q0 a 1 1 toy\nq Q0 b 1 0 toy\n", encoding="utf-8")
    elif problem == "missing_file":
        target = tmp_path / "absent.trec"
    elif problem == "encoding":
        target.write_bytes(b"\xff\xfe\x80")
    else:

        def fail(_):
            raise PolicyOptimizationError("test numerical failure; not infeasibility")

        monkeypatch.setattr(trial, "optimize_runs", fail)
    with pytest.raises(SystemExit) as error:
        trial.main(arguments(target, source, output))
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert captured.err.startswith("trial error:")
    assert "Traceback" not in captured.err
    if problem == "duplicate_rank":
        assert "target run: line 2" in captured.err
        assert "duplicate rank" in captured.err
    assert not output.exists()


def test_output_is_not_overwritten(tmp_path, capsys):
    target, source, output = inputs(tmp_path)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(SystemExit) as error:
        trial.main(arguments(target, source, output))
    assert error.value.code == 2
    assert "choose a new --output" in capsys.readouterr().err
    assert sentinel.read_text() == "untouched"


def test_unexpected_programming_error_is_not_disguised(tmp_path, monkeypatch):
    target, source, output = inputs(tmp_path)

    def fail(_):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(trial, "optimize_runs", fail)
    with pytest.raises(RuntimeError, match="unexpected bug"):
        trial.main(arguments(target, source, output))
