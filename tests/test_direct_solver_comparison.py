import pytest

from examples.direct_solver_comparison import compare, self_check


@pytest.mark.parametrize("support", [[], ["zero"], ["zero", "source"]])
@pytest.mark.parametrize("budget", [0.0, 0.1, 1.0])
def test_direct_and_public_paths_match_objective(support, budget):
    pytest.importorskip("scipy")
    context = {
        "target_ranking": ["zero", "fixed", "source", "tail"],
        "supported_items": support,
        "position_weights": [1.0, 0.6],
        "risk_budget": budget,
    }
    scores = {item: {"zero": 0.0, "source": 2.0}[item] for item in support}
    result = compare(context, scores)
    assert result["valid"]
    assert result["same_core_solver"]
    assert result["objective_abs_difference"] <= 1e-8
    assert result["paths"]["direct"]["regret"] <= budget + 1e-9


def test_packet_self_check_contains_all_nine_settings():
    pytest.importorskip("scipy")
    result = self_check()
    assert result["valid"]
    assert result["case_count"] == 9
    assert {(row["support_size"], row["budget"]) for row in result["cases"]} == {
        (support, budget) for support in (0, 1, 2) for budget in (0.0, 0.1, 1.0)
    }
