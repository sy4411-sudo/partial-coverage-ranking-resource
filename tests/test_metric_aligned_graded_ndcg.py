import itertools
import math
import random
import unittest

from metric_aligned_ranking.graded_ndcg_regret import (
    expected_graded_ndcg,
    graded_ndcg_regret_certificate,
)
from metric_aligned_ranking.ndcg_risk import (
    MetricAlignedRankingError,
    worst_case_binary_ndcg_regret,
)


def _ranking_exposure(
    ranking: tuple[int, ...],
    weights: tuple[float, ...],
) -> tuple[float, ...]:
    exposure = [0.0] * len(ranking)
    for position, item in enumerate(ranking):
        exposure[item] = weights[position]
    return tuple(exposure)


class GradedNdcgRegretTests(unittest.TestCase):
    def test_total_variation_is_the_tight_label_free_capacity_bound(self) -> None:
        weights = tuple(1.0 / math.log2(position + 2.0) for position in range(10))
        target = list(weights)
        for left, right in ((0, 1), (8, 9), (0, 9)):
            candidate = list(target)
            candidate[left], candidate[right] = (
                candidate[right],
                candidate[left],
            )
            total_variation = 0.5 * math.fsum(
                abs(target_value - candidate_value)
                for target_value, candidate_value in zip(target, candidate)
            )
            certificate = worst_case_binary_ndcg_regret(
                target,
                candidate,
                position_weights=weights,
            )
            self.assertAlmostEqual(certificate.regret, total_variation)

    def test_layer_cake_certificate_matches_direct_graded_ndcg(self) -> None:
        weights = (1.0, 0.6, 0.3)
        target = (1.0, 0.6, 0.3, 0.0)
        candidate = (0.6, 0.3, 1.0, 0.0)
        gains = (7.0, 1.0, 3.0, 0.0)

        certificate = graded_ndcg_regret_certificate(
            target,
            candidate,
            gains,
            position_weights=weights,
        )
        direct = expected_graded_ndcg(
            target,
            gains,
            position_weights=weights,
        ) - expected_graded_ndcg(
            candidate,
            gains,
            position_weights=weights,
        )

        self.assertAlmostEqual(certificate.regret, direct)
        self.assertAlmostEqual(
            sum(layer.convex_weight for layer in certificate.layers),
            1.0,
        )
        self.assertLessEqual(
            certificate.regret,
            certificate.worst_case_regret,
        )
        self.assertTrue(certificate.binary_adversaries_are_extremal)

    def test_binary_witness_attains_graded_supremum(self) -> None:
        weights = (1.0, 0.7, 0.4)
        target = (1.0, 0.7, 0.4, 0.0)
        candidate = (0.4, 1.0, 0.0, 0.7)
        binary = worst_case_binary_ndcg_regret(
            target,
            candidate,
            position_weights=weights,
        )
        witness = tuple(
            1.0 if item in binary.adversarial_relevant_items else 0.0
            for item in range(len(target))
        )
        graded = graded_ndcg_regret_certificate(
            target,
            candidate,
            witness,
            position_weights=weights,
        )

        self.assertAlmostEqual(graded.regret, binary.regret)
        self.assertAlmostEqual(graded.upper_bound_gap, 0.0)

    def test_exhaustive_small_integer_graded_gains_never_exceed_binary(self) -> None:
        weights = (1.0, 0.6, 0.3, 0.1)
        target = _ranking_exposure((0, 1, 2, 3), weights)
        for ranking in itertools.permutations(range(4)):
            candidate = _ranking_exposure(ranking, weights)
            binary = worst_case_binary_ndcg_regret(
                target,
                candidate,
                position_weights=weights,
            )
            maximum_graded = -math.inf
            for gains in itertools.product(range(4), repeat=4):
                if not any(gains):
                    continue
                certificate = graded_ndcg_regret_certificate(
                    target,
                    candidate,
                    gains,
                    position_weights=weights,
                )
                maximum_graded = max(maximum_graded, certificate.regret)
            self.assertAlmostEqual(maximum_graded, binary.regret)

    def test_random_stochastic_exposures_and_continuous_gains(self) -> None:
        rng = random.Random(20260727)
        weights = (1.0, 0.63, 0.5, 0.43)
        permutations = tuple(itertools.permutations(range(6), len(weights)))
        for _ in range(100):
            target_rankings = rng.sample(permutations, 4)
            candidate_rankings = rng.sample(permutations, 4)
            target_mass = [rng.random() for _ in target_rankings]
            candidate_mass = [rng.random() for _ in candidate_rankings]
            target_total = sum(target_mass)
            candidate_total = sum(candidate_mass)
            target = [0.0] * 6
            candidate = [0.0] * 6
            for ranking, mass in zip(target_rankings, target_mass):
                for position, item in enumerate(ranking):
                    target[item] += mass / target_total * weights[position]
            for ranking, mass in zip(candidate_rankings, candidate_mass):
                for position, item in enumerate(ranking):
                    candidate[item] += mass / candidate_total * weights[position]
            gains = tuple(rng.random() * rng.randrange(1, 5) for _ in range(6))
            certificate = graded_ndcg_regret_certificate(
                target,
                candidate,
                gains,
                position_weights=weights,
            )
            self.assertLessEqual(
                certificate.regret,
                certificate.worst_case_regret + 1e-12,
            )

    def test_rejects_zero_or_negative_gains(self) -> None:
        weights = (1.0, 0.5)
        with self.assertRaises(MetricAlignedRankingError):
            graded_ndcg_regret_certificate(
                (1.0, 0.5),
                (0.5, 1.0),
                (0.0, 0.0),
                position_weights=weights,
            )
        with self.assertRaises(MetricAlignedRankingError):
            graded_ndcg_regret_certificate(
                (1.0, 0.5),
                (0.5, 1.0),
                (1.0, -1.0),
                position_weights=weights,
            )


if __name__ == "__main__":
    unittest.main()
