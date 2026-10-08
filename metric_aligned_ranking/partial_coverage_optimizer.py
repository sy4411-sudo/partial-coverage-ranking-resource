"""Exact metric-aligned nDCG optimization on partial-coverage slots.

The target ranking in this module is deterministic.  Unsupported items are
fixed in place, while supported items may be randomized over the slots that
supported items occupy in the target ranking.  A doubly stochastic assignment
matrix represents that randomized policy.

The optimization maximizes a linear source-side DCG surrogate subject to an
exact worst-case binary-relevance nDCG-regret budget.  The exponentially many
relevance-set constraints are added by a cutting-plane oracle that sorts the
current exposure differences.  A Birkhoff--von Neumann decomposition turns the
solution into an explicit distribution over deployable rankings.

It imports SciPy lazily and does not add a mandatory project runtime dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .ndcg_risk import (
    worst_case_binary_ndcg_regret,
)


class RobustSlotRankingError(ValueError):
    """Raised when robust slot-ranking inputs or solver outputs are invalid."""


def _ranking(values: Sequence[int]) -> tuple[int, ...]:
    ranking = tuple(values)
    if not ranking or len(set(ranking)) != len(ranking):
        raise RobustSlotRankingError(
            "target_ranking must be non-empty and contain unique items"
        )
    if set(ranking) != set(range(len(ranking))):
        raise RobustSlotRankingError("target_ranking must be a permutation of range(n)")
    return ranking


def _supported_items(
    values: Sequence[int],
    *,
    item_count: int,
) -> tuple[int, ...]:
    support = tuple(values)
    if len(set(support)) != len(support):
        raise RobustSlotRankingError("supported_items must be unique")
    for item in support:
        if isinstance(item, bool) or not isinstance(item, int):
            raise RobustSlotRankingError("supported item ids must be integers")
        if not 0 <= item < item_count:
            raise RobustSlotRankingError("supported item id is out of range")
    return tuple(sorted(support))


def _position_weights(
    values: Sequence[float],
    *,
    item_count: int,
) -> tuple[float, ...]:
    weights: list[float] = []
    for index, raw_value in enumerate(values):
        if isinstance(raw_value, bool):
            raise RobustSlotRankingError(f"position_weights[{index}] must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise RobustSlotRankingError(
                f"position_weights[{index}] must be finite and positive"
            )
        if weights and value > weights[-1] + 1e-15:
            raise RobustSlotRankingError("position_weights must be non-increasing")
        weights.append(value)
    if not weights or len(weights) > item_count:
        raise RobustSlotRankingError(
            "position_weights length must lie in [1, item_count]"
        )
    return tuple(weights)


def _source_scores(
    values: Mapping[int, float],
    *,
    support: Sequence[int],
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for item in support:
        if item not in values or isinstance(values[item], bool):
            raise RobustSlotRankingError(
                "source_scores must provide a numeric score for every supported item"
            )
        score = float(values[item])
        if not math.isfinite(score):
            raise RobustSlotRankingError("source scores must be finite")
        scores[item] = score
    return scores


def _risk_budget(value: float) -> float:
    if isinstance(value, bool):
        raise RobustSlotRankingError("risk_budget must be numeric")
    budget = float(value)
    if not math.isfinite(budget) or budget < 0.0:
        raise RobustSlotRankingError("risk_budget must be finite and non-negative")
    return budget


def _clip_lp_unit_interval_residuals(
    values: Sequence[float], *, tolerance: float
) -> tuple[float, ...]:
    """Clip only bounded-LP residuals within a frozen feasibility tolerance."""

    output: list[float] = []
    for value in values:
        numeric = float(value)
        if (
            not math.isfinite(numeric)
            or numeric < -tolerance
            or numeric > 1 + tolerance
        ):
            raise RobustSlotRankingError(
                "LP assignment violates unit-interval bounds beyond tolerance"
            )
        if numeric < 0.0:
            numeric = 0.0
        elif numeric > 1.0:
            numeric = 1.0
        output.append(numeric)
    return tuple(output)


def _target_exposure(
    ranking: Sequence[int],
    position_weights: Sequence[float],
) -> tuple[float, ...]:
    exposure = [0.0] * len(ranking)
    for position, item in enumerate(ranking):
        if position < len(position_weights):
            exposure[item] = position_weights[position]
    return tuple(exposure)


def _assignment_exposure(
    *,
    target_exposure: Sequence[float],
    support: Sequence[int],
    support_slots: Sequence[int],
    slot_weights: Sequence[float],
    assignment: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    exposure = list(target_exposure)
    for item in support:
        exposure[item] = 0.0
    for row, item in enumerate(support):
        for column in range(len(support_slots)):
            exposure[item] += assignment[row][column] * slot_weights[column]
    return tuple(exposure)


@dataclass(frozen=True, slots=True)
class RankingMixtureComponent:
    """One deterministic full ranking in a deployable policy mixture."""

    probability: float
    ranking: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RobustSlotRankingResult:
    """Certified output of deterministic-baseline robust slot optimization."""

    assignment_matrix: tuple[tuple[float, ...], ...]
    ranking_mixture: tuple[RankingMixtureComponent, ...]
    target_exposure: tuple[float, ...]
    candidate_exposure: tuple[float, ...]
    exact_worst_case_binary_ndcg_regret: float
    risk_budget: float
    source_utility_target: float
    source_utility_candidate: float
    source_utility_gain: float
    cutting_plane_iterations: int
    active_relevance_constraints: tuple[tuple[int, ...], ...]
    deployment_policy_is_randomized: bool
    constraint_formulation: str = "cutting_plane"
    evaluation_only_pl_required: bool = False
    risk_exact_for_arbitrary_nonnegative_graded_gains: bool = True


def _birkhoff_decomposition(
    assignment: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RobustSlotRankingError(
            "SciPy is required for the exploratory Birkhoff decomposition"
        ) from exc

    matrix = np.asarray(assignment, dtype=float)
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise RobustSlotRankingError("assignment matrix must be square")
    if size == 0:
        return ((1.0, ()),)

    residual = matrix.copy()
    components: list[tuple[float, tuple[int, ...]]] = []
    threshold = max(1e-12, tolerance * 1e-3)
    for _ in range(size * size + 1):
        remaining_mass = float(residual.sum() / size)
        if remaining_mass <= tolerance:
            break
        cost = np.where(residual > threshold, 0.0, 1.0)
        rows, columns = linear_sum_assignment(cost)
        if float(cost[rows, columns].sum()) > 0.0:
            raise RobustSlotRankingError(
                "failed to find a positive perfect matching in assignment support"
            )
        coefficient = float(residual[rows, columns].min())
        if coefficient <= 0.0:
            raise RobustSlotRankingError(
                "Birkhoff decomposition selected a zero-mass edge"
            )
        permutation = [0] * size
        for row, column in zip(rows, columns):
            permutation[int(row)] = int(column)
            residual[row, column] -= coefficient
        residual[abs(residual) <= threshold] = 0.0
        components.append((coefficient, tuple(permutation)))
    else:  # pragma: no cover - defensive termination guard
        raise RobustSlotRankingError("Birkhoff decomposition did not terminate")

    total = math.fsum(probability for probability, _ in components)
    if total <= 0.0 or abs(total - 1.0) > 1e-7:
        raise RobustSlotRankingError("Birkhoff mixture probabilities do not sum to one")
    normalized = tuple(
        (probability / total, permutation)
        for probability, permutation in components
        if probability > tolerance
    )

    reconstructed = np.zeros_like(matrix)
    for probability, permutation in normalized:
        for row, column in enumerate(permutation):
            reconstructed[row, column] += probability
    if float(np.max(np.abs(reconstructed - matrix))) > 1e-7:
        raise RobustSlotRankingError(
            "Birkhoff mixture does not reconstruct the assignment matrix"
        )
    return normalized


def optimize_robust_slot_policy(
    target_ranking: Sequence[int],
    *,
    supported_items: Sequence[int],
    source_scores: Mapping[int, float],
    position_weights: Sequence[float],
    risk_budget: float,
    tolerance: float = 1e-9,
) -> RobustSlotRankingResult:
    """Optimize a deployable support-slot policy with an exact nDCG budget.

    The candidate list is the full target ranking.  ``position_weights`` may be
    shorter than the list, in which case positions after the cutoff receive
    zero exposure.  Supported items can therefore move into or out of top-k,
    but unsupported items never change position in any mixture component.
    """

    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RobustSlotRankingError(
            "SciPy is required for the exploratory cutting-plane solver"
        ) from exc

    ranking = _ranking(target_ranking)
    support = _supported_items(supported_items, item_count=len(ranking))
    weights = _position_weights(position_weights, item_count=len(ranking))
    scores = _source_scores(source_scores, support=support)
    budget = _risk_budget(risk_budget)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1e-3:
        raise RobustSlotRankingError("tolerance must lie in (0, 1e-3)")

    target_exposure = _target_exposure(ranking, weights)
    support_slots = tuple(
        position for position, item in enumerate(ranking) if item in support
    )
    size = len(support)
    slot_weights = tuple(
        weights[position] if position < len(weights) else 0.0
        for position in support_slots
    )
    target_utility = math.fsum(scores[item] * target_exposure[item] for item in support)

    if size <= 1:
        matrix = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(size))
            for row in range(size)
        )
        return RobustSlotRankingResult(
            assignment_matrix=matrix,
            ranking_mixture=(RankingMixtureComponent(1.0, ranking),),
            target_exposure=target_exposure,
            candidate_exposure=target_exposure,
            exact_worst_case_binary_ndcg_regret=0.0,
            risk_budget=budget,
            source_utility_target=target_utility,
            source_utility_candidate=target_utility,
            source_utility_gain=0.0,
            cutting_plane_iterations=0,
            active_relevance_constraints=(),
            deployment_policy_is_randomized=False,
        )

    variable_count = size * size
    objective = np.zeros(variable_count, dtype=float)
    for row, item in enumerate(support):
        for column, slot_weight in enumerate(slot_weights):
            objective[row * size + column] = -scores[item] * slot_weight

    equalities: list[list[float]] = []
    equality_rhs: list[float] = []
    for row in range(size):
        constraint = [0.0] * variable_count
        for column in range(size):
            constraint[row * size + column] = 1.0
        equalities.append(constraint)
        equality_rhs.append(1.0)
    for column in range(size):
        constraint = [0.0] * variable_count
        for row in range(size):
            constraint[row * size + column] = 1.0
        equalities.append(constraint)
        equality_rhs.append(1.0)

    inequalities: list[list[float]] = []
    inequality_rhs: list[float] = []
    active_sets: list[tuple[int, ...]] = []
    active_set_keys: set[tuple[int, ...]] = set()
    max_iterations = 2 ** len(ranking)
    assignment_array: object | None = None
    candidate_exposure: tuple[float, ...] | None = None
    exact_regret = math.inf

    for iteration in range(1, max_iterations + 1):
        result = linprog(
            objective,
            A_ub=np.asarray(inequalities) if inequalities else None,
            b_ub=np.asarray(inequality_rhs) if inequality_rhs else None,
            A_eq=np.asarray(equalities),
            b_eq=np.asarray(equality_rhs),
            bounds=(0.0, 1.0),
            method="highs",
        )
        if not result.success:
            raise RobustSlotRankingError(f"cutting-plane LP failed: {result.message}")
        assignment_array = result.x.reshape((size, size))
        assignment = tuple(
            tuple(float(assignment_array[row, column]) for column in range(size))
            for row in range(size)
        )
        candidate_exposure = _assignment_exposure(
            target_exposure=target_exposure,
            support=support,
            support_slots=support_slots,
            slot_weights=slot_weights,
            assignment=assignment,
        )
        certificate = worst_case_binary_ndcg_regret(
            target_exposure,
            candidate_exposure,
            position_weights=weights,
        )
        exact_regret = certificate.regret
        if exact_regret <= budget + tolerance:
            break

        adversarial_set = certificate.adversarial_relevant_items
        if adversarial_set in active_set_keys:
            raise RobustSlotRankingError(
                "separation oracle repeated a violated relevance constraint"
            )
        active_set_keys.add(adversarial_set)
        active_sets.append(adversarial_set)
        relevant = frozenset(adversarial_set)
        relevant_count = len(relevant)
        ideal_dcg = math.fsum(weights[: min(relevant_count, len(weights))])
        target_support_mass = math.fsum(
            target_exposure[item] for item in support if item in relevant
        )
        constraint = [0.0] * variable_count
        for row, item in enumerate(support):
            if item not in relevant:
                continue
            for column, slot_weight in enumerate(slot_weights):
                constraint[row * size + column] = -slot_weight
        inequalities.append(constraint)
        inequality_rhs.append(budget * ideal_dcg - target_support_mass)
    else:  # pragma: no cover - finite-constraint termination guard
        raise RobustSlotRankingError("cutting-plane solver did not terminate")

    if assignment_array is None or candidate_exposure is None:
        raise RobustSlotRankingError("cutting-plane solver returned no policy")
    assignment = tuple(
        tuple(float(assignment_array[row, column]) for column in range(size))
        for row in range(size)
    )
    decomposition = _birkhoff_decomposition(assignment, tolerance=tolerance)
    mixture: list[RankingMixtureComponent] = []
    for probability, permutation in decomposition:
        candidate = list(ranking)
        for row, item in enumerate(support):
            candidate[support_slots[permutation[row]]] = item
        mixture.append(
            RankingMixtureComponent(
                probability=probability,
                ranking=tuple(candidate),
            )
        )

    candidate_utility = math.fsum(
        scores[item] * candidate_exposure[item] for item in support
    )
    randomized = sum(component.probability > tolerance for component in mixture) > 1
    return RobustSlotRankingResult(
        assignment_matrix=assignment,
        ranking_mixture=tuple(mixture),
        target_exposure=target_exposure,
        candidate_exposure=candidate_exposure,
        exact_worst_case_binary_ndcg_regret=exact_regret,
        risk_budget=budget,
        source_utility_target=target_utility,
        source_utility_candidate=candidate_utility,
        source_utility_gain=candidate_utility - target_utility,
        cutting_plane_iterations=iteration,
        active_relevance_constraints=tuple(active_sets),
        deployment_policy_is_randomized=randomized,
    )


def optimize_robust_slot_policy_compact(
    target_ranking: Sequence[int],
    *,
    supported_items: Sequence[int],
    source_scores: Mapping[int, float],
    position_weights: Sequence[float],
    risk_budget: float,
    tolerance: float = 1e-9,
) -> RobustSlotRankingResult:
    """Solve the same policy problem with a compact top-sum extended LP.

    For the support-coordinate exposure loss ``z = e_target - e_candidate``,
    unsupported coordinates are zero and ``sum(z) = 0``.  Exact worst-case
    binary-nDCG regret is therefore bounded by ``risk_budget`` exactly when

    ``sum_largest_r(z) <= risk_budget * ideal_dcg(r)``

    for every ``r`` from one through ``|support| - 1``.  Each top-r sum has
    the standard linear epigraph

    ``min_eta r * eta + sum_i max(z_i - eta, 0)``.

    Only supported slots above the evaluation cutoff can have positive
    exposure.  Writing their count as ``k_s``, positive exposure loss can
    occur on at most ``k_s`` items.  A rectangular matching formulation
    therefore needs O(|support| * k_s) variables and nonzeros, rather than a
    square assignment or exponentially many relevance-set constraints.
    Binary relevance vectors are extremal for arbitrary non-negative graded
    gains, so this is also an exact graded-nDCG trust region rather than a
    binary-label special case.
    """

    try:
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RobustSlotRankingError(
            "SciPy is required for the exploratory compact solver"
        ) from exc

    ranking = _ranking(target_ranking)
    support = _supported_items(supported_items, item_count=len(ranking))
    weights = _position_weights(position_weights, item_count=len(ranking))
    scores = _source_scores(source_scores, support=support)
    budget = _risk_budget(risk_budget)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1e-3:
        raise RobustSlotRankingError("tolerance must lie in (0, 1e-3)")

    target_exposure = _target_exposure(ranking, weights)
    support_slots = tuple(
        position for position, item in enumerate(ranking) if item in support
    )
    size = len(support)
    slot_weights = tuple(
        weights[position] if position < len(weights) else 0.0
        for position in support_slots
    )
    target_utility = math.fsum(scores[item] * target_exposure[item] for item in support)

    if size <= 1:
        matrix = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(size))
            for row in range(size)
        )
        return RobustSlotRankingResult(
            assignment_matrix=matrix,
            ranking_mixture=(RankingMixtureComponent(1.0, ranking),),
            target_exposure=target_exposure,
            candidate_exposure=target_exposure,
            exact_worst_case_binary_ndcg_regret=0.0,
            risk_budget=budget,
            source_utility_target=target_utility,
            source_utility_candidate=target_utility,
            source_utility_gain=0.0,
            cutting_plane_iterations=0,
            active_relevance_constraints=(),
            deployment_policy_is_randomized=False,
            constraint_formulation="compact_top_k_matching",
        )

    active_slot_count = sum(slot_weight > 0.0 for slot_weight in slot_weights)
    if active_slot_count == 0:
        target_assignment = tuple(
            tuple(
                1.0 if ranking[support_slots[column]] == item else 0.0
                for column in range(size)
            )
            for item in support
        )
        return RobustSlotRankingResult(
            assignment_matrix=target_assignment,
            ranking_mixture=(RankingMixtureComponent(1.0, ranking),),
            target_exposure=target_exposure,
            candidate_exposure=target_exposure,
            exact_worst_case_binary_ndcg_regret=0.0,
            risk_budget=budget,
            source_utility_target=target_utility,
            source_utility_candidate=target_utility,
            source_utility_gain=0.0,
            cutting_plane_iterations=0,
            active_relevance_constraints=(),
            deployment_policy_is_randomized=False,
            constraint_formulation="compact_top_k_matching",
        )

    active_slot_weights = slot_weights[:active_slot_count]
    if any(slot_weight != 0.0 for slot_weight in slot_weights[active_slot_count:]):
        raise RobustSlotRankingError("positive support-slot weights must form a prefix")

    assignment_variable_count = size * active_slot_count
    relevant_count_total = min(active_slot_count, size - 1)
    loss_offset = assignment_variable_count
    eta_offset = loss_offset + size
    slack_offset = eta_offset + relevant_count_total
    variable_count = (
        assignment_variable_count
        + size
        + relevant_count_total
        + relevant_count_total * size
    )

    objective = np.zeros(variable_count, dtype=float)
    for row, item in enumerate(support):
        for column, slot_weight in enumerate(active_slot_weights):
            objective[row * active_slot_count + column] = -scores[item] * slot_weight

    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_values: list[float] = []
    equality_rhs: list[float] = []
    for column in range(active_slot_count):
        constraint_index = len(equality_rhs)
        for row in range(size):
            equality_rows.append(constraint_index)
            equality_columns.append(row * active_slot_count + column)
            equality_values.append(1.0)
        equality_rhs.append(1.0)
    for row, item in enumerate(support):
        constraint_index = len(equality_rhs)
        for column, slot_weight in enumerate(active_slot_weights):
            equality_rows.append(constraint_index)
            equality_columns.append(row * active_slot_count + column)
            equality_values.append(slot_weight)
        equality_rows.append(constraint_index)
        equality_columns.append(loss_offset + row)
        equality_values.append(1.0)
        equality_rhs.append(target_exposure[item])

    inequality_rows: list[int] = []
    inequality_columns: list[int] = []
    inequality_values: list[float] = []
    inequality_rhs: list[float] = []
    for row in range(size):
        constraint_index = len(inequality_rhs)
        for column in range(active_slot_count):
            inequality_rows.append(constraint_index)
            inequality_columns.append(row * active_slot_count + column)
            inequality_values.append(1.0)
        inequality_rhs.append(1.0)

    for relevant_count in range(1, relevant_count_total + 1):
        eta_index = eta_offset + relevant_count - 1
        ideal_dcg = math.fsum(weights[: min(relevant_count, len(weights))])

        constraint_index = len(inequality_rhs)
        inequality_rows.append(constraint_index)
        inequality_columns.append(eta_index)
        inequality_values.append(float(relevant_count))
        for row in range(size):
            inequality_rows.append(constraint_index)
            inequality_columns.append(slack_offset + (relevant_count - 1) * size + row)
            inequality_values.append(1.0)
        inequality_rhs.append(budget * ideal_dcg)

        for row in range(size):
            constraint_index = len(inequality_rhs)
            inequality_rows.extend((constraint_index,) * 3)
            inequality_columns.extend(
                (
                    loss_offset + row,
                    eta_index,
                    slack_offset + (relevant_count - 1) * size + row,
                )
            )
            inequality_values.extend((1.0, -1.0, -1.0))
            inequality_rhs.append(0.0)

    equality_matrix = coo_matrix(
        (equality_values, (equality_rows, equality_columns)),
        shape=(len(equality_rhs), variable_count),
    ).tocsr()
    inequality_matrix = coo_matrix(
        (inequality_values, (inequality_rows, inequality_columns)),
        shape=(len(inequality_rhs), variable_count),
    ).tocsr()

    bounds = (
        [(0.0, 1.0)] * assignment_variable_count
        + [(None, None)] * size
        + [(None, None)] * relevant_count_total
        + [(0.0, None)] * (relevant_count_total * size)
    )
    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=np.asarray(inequality_rhs),
        A_eq=equality_matrix,
        b_eq=np.asarray(equality_rhs),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RobustSlotRankingError(f"compact top-sum LP failed: {result.message}")

    active_assignment = np.asarray(
        _clip_lp_unit_interval_residuals(
            result.x[:assignment_variable_count],
            tolerance=max(1e-7, tolerance),
        )
    ).reshape((size, active_slot_count))
    candidate_exposure_values = list(target_exposure)
    for row, item in enumerate(support):
        candidate_exposure_values[item] = math.fsum(
            float(active_assignment[row, column]) * slot_weight
            for column, slot_weight in enumerate(active_slot_weights)
        )
    candidate_exposure = tuple(candidate_exposure_values)

    # Complete the rectangular top-k matching marginals with zero-exposure
    # support slots.  The result is doubly stochastic and can therefore use
    # the same exact Birkhoff deployment path as the square formulation.
    assignment_array = np.zeros((size, size), dtype=float)
    assignment_array[:, :active_slot_count] = active_assignment
    residual_row_mass = [
        max(0.0, 1.0 - float(active_assignment[row, :].sum())) for row in range(size)
    ]
    row_cursor = 0
    for column in range(active_slot_count, size):
        remaining_column_mass = 1.0
        while remaining_column_mass > tolerance:
            while row_cursor < size and residual_row_mass[row_cursor] <= tolerance:
                row_cursor += 1
            if row_cursor >= size:
                raise RobustSlotRankingError(
                    "failed to complete rectangular assignment"
                )
            allocated = min(
                remaining_column_mass,
                residual_row_mass[row_cursor],
            )
            assignment_array[row_cursor, column] = allocated
            residual_row_mass[row_cursor] -= allocated
            remaining_column_mass -= allocated
    if max(residual_row_mass, default=0.0) > 1e-7:
        raise RobustSlotRankingError("rectangular assignment completion lost mass")

    assignment = tuple(
        tuple(float(assignment_array[row, column]) for column in range(size))
        for row in range(size)
    )
    exact_regret = worst_case_binary_ndcg_regret(
        target_exposure,
        candidate_exposure,
        position_weights=weights,
    ).regret
    if exact_regret > budget + tolerance:
        raise RobustSlotRankingError(
            "compact top-sum LP returned a policy outside the exact risk budget"
        )

    decomposition = _birkhoff_decomposition(assignment, tolerance=tolerance)
    mixture: list[RankingMixtureComponent] = []
    for probability, permutation in decomposition:
        candidate = list(ranking)
        for row, item in enumerate(support):
            candidate[support_slots[permutation[row]]] = item
        mixture.append(
            RankingMixtureComponent(
                probability=probability,
                ranking=tuple(candidate),
            )
        )

    candidate_utility = math.fsum(
        scores[item] * candidate_exposure[item] for item in support
    )
    randomized = sum(component.probability > tolerance for component in mixture) > 1
    return RobustSlotRankingResult(
        assignment_matrix=assignment,
        ranking_mixture=tuple(mixture),
        target_exposure=target_exposure,
        candidate_exposure=candidate_exposure,
        exact_worst_case_binary_ndcg_regret=exact_regret,
        risk_budget=budget,
        source_utility_target=target_utility,
        source_utility_candidate=candidate_utility,
        source_utility_gain=candidate_utility - target_utility,
        cutting_plane_iterations=0,
        active_relevance_constraints=(),
        deployment_policy_is_randomized=randomized,
        constraint_formulation="compact_top_k_matching",
    )
