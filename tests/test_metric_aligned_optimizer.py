import importlib.util
import itertools
import math
import random
import unittest

from metric_aligned_ranking.ndcg_risk import (
    ranking_exposure,
    worst_case_binary_ndcg_regret,
)
from metric_aligned_ranking.partial_coverage_optimizer import (
    RobustSlotRankingError,
    _clip_lp_unit_interval_residuals,
    optimize_robust_slot_policy,
    optimize_robust_slot_policy_compact,
)

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(SCIPY_AVAILABLE, "exploratory solver requires scipy")
class RobustSlotRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ranking = (0, 2, 1, 3)
        self.support = (0, 1)
        self.scores = {0: 0.1, 1: 1.0}
        self.weights = (1.0, 1.0 / math.log2(3.0))

    def test_lp_bound_residuals_are_clipped_but_larger_violations_fail(self) -> None:
        self.assertEqual(
            _clip_lp_unit_interval_residuals((-5e-8, 0.5, 1.0 + 5e-8), tolerance=1e-7),
            (0.0, 0.5, 1.0),
        )
        with self.assertRaisesRegex(RobustSlotRankingError, "beyond tolerance"):
            _clip_lp_unit_interval_residuals((-2e-7,), tolerance=1e-7)

    def test_unconstrained_endpoint_moves_supported_item_across_cutoff(self) -> None:
        result = optimize_robust_slot_policy(
            self.ranking,
            supported_items=self.support,
            source_scores=self.scores,
            position_weights=self.weights,
            risk_budget=1.0,
        )
        self.assertEqual(len(result.ranking_mixture), 1)
        self.assertEqual(result.ranking_mixture[0].ranking, (1, 2, 0, 3))
        self.assertGreater(result.source_utility_gain, 0.0)
        self.assertFalse(result.evaluation_only_pl_required)

    def test_every_mixture_component_fixes_unsupported_positions(self) -> None:
        result = optimize_robust_slot_policy(
            self.ranking,
            supported_items=self.support,
            source_scores=self.scores,
            position_weights=self.weights,
            risk_budget=0.1,
        )
        for component in result.ranking_mixture:
            self.assertAlmostEqual(
                sum(item.probability for item in result.ranking_mixture),
                1.0,
            )
            self.assertEqual(component.ranking[1], 2)
            self.assertEqual(component.ranking[3], 3)

    def test_intermediate_budget_is_tight_and_requires_randomization(self) -> None:
        result = optimize_robust_slot_policy(
            self.ranking,
            supported_items=self.support,
            source_scores=self.scores,
            position_weights=self.weights,
            risk_budget=0.1,
        )
        self.assertAlmostEqual(
            result.exact_worst_case_binary_ndcg_regret,
            0.1,
        )
        self.assertTrue(result.deployment_policy_is_randomized)
        self.assertGreater(result.source_utility_gain, 0.0)
        self.assertGreaterEqual(len(result.active_relevance_constraints), 1)
        self.assertTrue(result.risk_exact_for_arbitrary_nonnegative_graded_gains)

    def test_zero_budget_cannot_change_exposure(self) -> None:
        result = optimize_robust_slot_policy(
            self.ranking,
            supported_items=self.support,
            source_scores=self.scores,
            position_weights=self.weights,
            risk_budget=0.0,
        )
        for actual, expected in zip(
            result.candidate_exposure,
            result.target_exposure,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.source_utility_gain, 0.0)

    def test_birkhoff_mixture_reconstructs_candidate_exposure(self) -> None:
        result = optimize_robust_slot_policy(
            self.ranking,
            supported_items=self.support,
            source_scores=self.scores,
            position_weights=self.weights,
            risk_budget=0.1,
        )
        distribution = {
            component.ranking[: len(self.weights)]: component.probability
            for component in result.ranking_mixture
        }
        reconstructed = ranking_exposure(
            distribution,
            item_count=len(self.ranking),
            position_weights=self.weights,
        )
        for actual, expected in zip(
            reconstructed,
            result.candidate_exposure,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_cutting_plane_matches_all_constraints_permutation_lp(self) -> None:
        import numpy as np
        from scipy.optimize import linprog

        ranking = (0, 1, 2, 3)
        support = (0, 1, 2)
        scores = {0: 0.1, 1: 0.7, 2: 1.0}
        weights = (1.0, 0.6, 0.3)
        budget = 0.08
        result = optimize_robust_slot_policy(
            ranking,
            supported_items=support,
            source_scores=scores,
            position_weights=weights,
            risk_budget=budget,
        )

        permutations = tuple(itertools.permutations(support))
        exposures = []
        utilities = []
        for permutation in permutations:
            exposure = [0.0] * len(ranking)
            for position, item in enumerate(permutation):
                exposure[item] = weights[position]
            exposures.append(tuple(exposure))
            utilities.append(sum(scores[item] * exposure[item] for item in support))
        target_exposure = exposures[0]

        constraints = []
        rhs = []
        for count in range(1, len(ranking) + 1):
            ideal_dcg = sum(weights[: min(count, len(weights))])
            for relevant in itertools.combinations(range(len(ranking)), count):
                constraints.append(
                    [
                        -sum(exposure[item] for item in relevant)
                        for exposure in exposures
                    ]
                )
                rhs.append(
                    budget * ideal_dcg - sum(target_exposure[item] for item in relevant)
                )
        oracle = linprog(
            -np.asarray(utilities),
            A_ub=np.asarray(constraints),
            b_ub=np.asarray(rhs),
            A_eq=np.ones((1, len(permutations))),
            b_eq=np.ones(1),
            bounds=(0.0, 1.0),
            method="highs",
        )
        self.assertTrue(oracle.success)
        self.assertAlmostEqual(
            result.source_utility_candidate,
            -float(oracle.fun),
        )
        certificate = worst_case_binary_ndcg_regret(
            target_exposure,
            result.candidate_exposure,
            position_weights=weights,
        )
        self.assertLessEqual(certificate.regret, budget + 1e-9)

    def test_compact_top_k_lp_matches_cutting_plane_on_random_instances(
        self,
    ) -> None:
        rng = random.Random(947271)
        for item_count in range(3, 7):
            for _ in range(3):
                ranking_list = list(range(item_count))
                rng.shuffle(ranking_list)
                ranking = tuple(ranking_list)
                support_size = rng.randint(2, item_count)
                support = tuple(sorted(rng.sample(range(item_count), support_size)))
                scores = {item: rng.random() for item in support}
                cutoff = rng.randint(1, item_count)
                weights = tuple(
                    1.0 / math.log2(position + 2.0) for position in range(cutoff)
                )
                budget = rng.uniform(0.0, 0.25)

                cutting_plane = optimize_robust_slot_policy(
                    ranking,
                    supported_items=support,
                    source_scores=scores,
                    position_weights=weights,
                    risk_budget=budget,
                )
                compact = optimize_robust_slot_policy_compact(
                    ranking,
                    supported_items=support,
                    source_scores=scores,
                    position_weights=weights,
                    risk_budget=budget,
                )

                self.assertAlmostEqual(
                    compact.source_utility_candidate,
                    cutting_plane.source_utility_candidate,
                    places=8,
                )
                self.assertLessEqual(
                    compact.exact_worst_case_binary_ndcg_regret,
                    budget + 1e-8,
                )
                self.assertEqual(
                    compact.constraint_formulation,
                    "compact_top_k_matching",
                )

    def test_full_robust_polytope_can_strictly_beat_scalar_endpoint_mix(
        self,
    ) -> None:
        ranking = (1, 2, 4, 0, 3)
        support = tuple(range(5))
        scores = {0: 0.019, 1: 0.053, 2: 0.245, 3: 0.830, 4: 0.118}
        weights = tuple(1.0 / math.log2(position + 2.0) for position in range(5))
        budget = 0.1
        result = optimize_robust_slot_policy(
            ranking,
            supported_items=support,
            source_scores=scores,
            position_weights=weights,
            risk_budget=budget,
        )

        source_endpoint = tuple(sorted(support, key=lambda item: (-scores[item], item)))
        endpoint_exposure = [0.0] * len(ranking)
        for position, item in enumerate(source_endpoint):
            endpoint_exposure[item] = weights[position]
        unit_regret = worst_case_binary_ndcg_regret(
            result.target_exposure,
            endpoint_exposure,
            position_weights=weights,
        ).regret
        alpha = min(1.0, budget / unit_regret)
        endpoint_utility = sum(
            scores[item] * endpoint_exposure[item] for item in support
        )
        scalar_utility = (
            1.0 - alpha
        ) * result.source_utility_target + alpha * endpoint_utility

        self.assertGreater(
            result.source_utility_candidate - scalar_utility,
            0.10,
        )

    def test_expected_policy_budget_does_not_bound_every_realized_ranking(
        self,
    ) -> None:
        ranking = (1, 2, 4, 0, 3)
        support = tuple(range(5))
        scores = {0: 0.019, 1: 0.053, 2: 0.245, 3: 0.830, 4: 0.118}
        weights = tuple(1.0 / math.log2(position + 2.0) for position in range(5))
        budget = 0.1
        result = optimize_robust_slot_policy(
            ranking,
            supported_items=support,
            source_scores=scores,
            position_weights=weights,
            risk_budget=budget,
        )

        component_regrets = []
        for component in result.ranking_mixture:
            component_exposure = ranking_exposure(
                {component.ranking: 1.0},
                item_count=len(ranking),
                position_weights=weights,
            )
            component_regrets.append(
                worst_case_binary_ndcg_regret(
                    result.target_exposure,
                    component_exposure,
                    position_weights=weights,
                ).regret
            )

        self.assertLessEqual(
            result.exact_worst_case_binary_ndcg_regret,
            budget + 1e-9,
        )
        self.assertGreater(max(component_regrets), 3.0 * budget)

    def test_single_supported_item_is_inactive(self) -> None:
        result = optimize_robust_slot_policy(
            self.ranking,
            supported_items=(0,),
            source_scores={0: 1.0},
            position_weights=self.weights,
            risk_budget=1.0,
        )
        self.assertEqual(result.ranking_mixture[0].ranking, self.ranking)
        self.assertEqual(result.source_utility_gain, 0.0)
        self.assertFalse(result.deployment_policy_is_randomized)

    def test_randomized_small_instances_are_feasible_and_reconstructable(
        self,
    ) -> None:
        rng = random.Random(20260727)
        for item_count in range(3, 7):
            for _ in range(4):
                ranking_list = list(range(item_count))
                rng.shuffle(ranking_list)
                ranking = tuple(ranking_list)
                support_size = rng.randint(2, min(4, item_count))
                support = tuple(sorted(rng.sample(range(item_count), support_size)))
                scores = {item: rng.random() for item in support}
                cutoff = rng.randint(1, item_count)
                weights = tuple(
                    1.0 / math.log2(position + 2.0) for position in range(cutoff)
                )
                budget = rng.uniform(0.0, 0.3)

                result = optimize_robust_slot_policy(
                    ranking,
                    supported_items=support,
                    source_scores=scores,
                    position_weights=weights,
                    risk_budget=budget,
                )
                self.assertLessEqual(
                    result.exact_worst_case_binary_ndcg_regret,
                    budget + 1e-8,
                )
                self.assertAlmostEqual(
                    sum(component.probability for component in result.ranking_mixture),
                    1.0,
                )
                support_set = frozenset(support)
                for component in result.ranking_mixture:
                    for position, item in enumerate(ranking):
                        if item not in support_set:
                            self.assertEqual(component.ranking[position], item)

                distribution: dict[tuple[int, ...], float] = {}
                for component in result.ranking_mixture:
                    top_k = component.ranking[:cutoff]
                    distribution[top_k] = (
                        distribution.get(top_k, 0.0) + component.probability
                    )
                reconstructed = ranking_exposure(
                    distribution,
                    item_count=item_count,
                    position_weights=weights,
                )
                for actual, expected in zip(
                    reconstructed,
                    result.candidate_exposure,
                ):
                    self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
