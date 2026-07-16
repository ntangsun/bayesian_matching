#!/usr/bin/env python3
"""
matching_core.py

Small matching utilities shared by the diagnostic code.

Why this file exists:
- The diagnostic runner should measure the sampler itself, not the regression
  and variance-estimation work in run_matching_from_datasets.py.
- This file therefore contains only low-level matching pieces: distances,
  feasible initial matchings, simple matching summaries, and one local
  constrained 1-to-K Metropolis-Hastings proposal step.

Important convention:
- dist has shape (n_r, n_d).
- match_cols has shape (n_r, K).
- match_cols[i, :] are donor-column indices assigned to recipient i.
- Donors are never reused in constrained matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def recipient_donor_indices(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row indices for recipients (g=1) and donors (g=0)."""
    rc = np.where(g == 1)[0]
    dn = np.where(g == 0)[0]
    return rc, dn


def split_recipient_donor_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split full X/Y arrays into recipient and donor arrays."""
    rc, dn = recipient_donor_indices(g)
    return X[rc], X[dn], Y[rc], Y[dn]


def distance_matrix_from_groups(
    X_rc: np.ndarray,
    X_dn: np.ndarray,
    dist_type: int = 2,
) -> np.ndarray:
    """
    Compute recipient-by-donor distances.

    dist_type=2 is Euclidean distance and is the default used by the existing
    matching scripts.
    """
    if dist_type == 1:
        return np.abs(X_rc[:, None, :] - X_dn[None, :, :]).sum(axis=2)

    if dist_type == 2:
        return np.sqrt(((X_rc[:, None, :] - X_dn[None, :, :]) ** 2).sum(axis=2))

    if dist_type == 3:
        X = np.vstack([X_rc, X_dn])
        cov = np.cov(X, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        diff = X_rc[:, None, :] - X_dn[None, :, :]
        return np.einsum("...k,kl,...l->...", diff, inv_cov, diff)

    raise ValueError("dist_type must be 1, 2, or 3")


def init_constrained_greedy(dist: np.ndarray, k: int, largest: bool = False) -> np.ndarray:
    """
    Greedy feasible matching without donor reuse.

    largest=False gives a nearest-distance start.
    largest=True gives a deliberately high-distance start.
    """
    n_r, _ = dist.shape
    match_cols = np.empty((n_r, k), dtype=int)
    used: set[int] = set()

    for slot in range(k):
        for i in range(n_r):
            order = np.argsort(dist[i])
            if largest:
                order = order[::-1]
            for donor_col in order:
                donor_col = int(donor_col)
                if donor_col not in used:
                    match_cols[i, slot] = donor_col
                    used.add(donor_col)
                    break
            else:
                raise RuntimeError("Could not construct a feasible greedy matching.")

    return match_cols


def init_constrained_random(n_r: int, n_d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random feasible matching without donor reuse.

    We shuffle donor columns, take the first n_r*K donors, and reshape them
    into recipient rows.
    """
    if n_r * k > n_d:
        raise ValueError("Need n_d >= n_r*K for constrained matching.")
    donors = rng.permutation(n_d)[: n_r * k]
    return donors.reshape(n_r, k)


def unmatched_donors(match_cols: np.ndarray, n_d: int) -> np.ndarray:
    """Return donor columns not currently assigned to any recipient."""
    used = np.zeros(n_d, dtype=bool)
    used[match_cols.ravel()] = True
    return np.where(~used)[0]


def perturb_matching(
    match_cols: np.ndarray,
    n_d: int,
    rng: np.random.Generator,
    n_steps: int,
) -> np.ndarray:
    """
    Make a feasible matching different from its starting point.

    This is used for a dispersed "perturbed greedy" chain start. It changes
    the initial matching without using the posterior target distribution.
    """
    out = match_cols.copy()
    n_r, k = out.shape
    unmatched = unmatched_donors(out, n_d)

    for _ in range(n_steps):
        use_unmatched = len(unmatched) > 0 and rng.random() < 0.5

        if use_unmatched:
            row = int(rng.integers(n_r))
            slot = int(rng.integers(k))
            unmatched_pos = int(rng.integers(len(unmatched)))
            old = out[row, slot]
            out[row, slot] = unmatched[unmatched_pos]
            unmatched[unmatched_pos] = old
        else:
            row1 = int(rng.integers(n_r))
            row2 = int(rng.integers(n_r - 1))
            if row2 >= row1:
                row2 += 1
            slot1 = int(rng.integers(k))
            slot2 = int(rng.integers(k))
            out[row1, slot1], out[row2, slot2] = out[row2, slot2], out[row1, slot1]

    return out


def initial_matching(
    start_kind: str,
    dist: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build one of the named dispersed starting matchings for diagnostics.

    Supported starts:
    - greedy: low-distance deterministic start
    - random: random feasible start
    - high_distance: deliberately high-distance deterministic start
    - perturbed_greedy: greedy start after many random feasible edits
    """
    n_r, n_d = dist.shape

    if start_kind == "greedy":
        return init_constrained_greedy(dist, k=k, largest=False)

    if start_kind == "random":
        return init_constrained_random(n_r, n_d, k, rng)

    if start_kind == "high_distance":
        return init_constrained_greedy(dist, k=k, largest=True)

    if start_kind == "perturbed_greedy":
        base = init_constrained_greedy(dist, k=k, largest=False)
        return perturb_matching(base, n_d=n_d, rng=rng, n_steps=10 * n_r * k)

    raise ValueError(f"Unknown start_kind={start_kind!r}")


def total_distance(dist: np.ndarray, match_cols: np.ndarray) -> float:
    """Total matching distance D(C)."""
    rows = np.arange(match_cols.shape[0])[:, None]
    return float(dist[rows, match_cols].sum())


def row_retained_count(row_values: np.ndarray, initial_row_set: set[int]) -> int:
    """How many current donors for one recipient were also in its initial row."""
    return sum(int(donor) in initial_row_set for donor in row_values)


@dataclass
class ConstrainedOneToKState:
    """
    Mutable state for the constrained 1-to-K diagnostic sampler.

    The cached sums make thin=1 practical:
    - d_total is updated by distance deltas.
    - matched_y_sum and matched_x_sum change only when a matched donor is
      replaced by an unmatched donor.
    - retained_count is updated only for the recipient rows touched by an
      accepted proposal.
    """

    match_cols: np.ndarray
    unmatched: np.ndarray
    d_total: float
    matched_y_sum: float
    matched_x_sum: np.ndarray
    initial_sets: list[set[int]]
    retained_count: int


def make_constrained_state(
    match_cols: np.ndarray,
    dist: np.ndarray,
    X_dn: np.ndarray,
    Y_dn: np.ndarray,
) -> ConstrainedOneToKState:
    """Create a state object and cache the summaries needed for tracing."""
    n_r, k = match_cols.shape
    n_d = dist.shape[1]
    match_cols = match_cols.copy()
    initial_sets = [set(int(x) for x in match_cols[i]) for i in range(n_r)]

    matched = match_cols.ravel()
    retained_count = n_r * k

    return ConstrainedOneToKState(
        match_cols=match_cols,
        unmatched=unmatched_donors(match_cols, n_d=n_d),
        d_total=total_distance(dist, match_cols),
        matched_y_sum=float(Y_dn[matched].sum()),
        matched_x_sum=X_dn[matched].sum(axis=0),
        initial_sets=initial_sets,
        retained_count=retained_count,
    )


def state_tau(state: ConstrainedOneToKState, y_rc_mean: float) -> float:
    """Simple matching treatment-effect trace: mean(Y_rc) - mean(Y_matched_dn)."""
    n_assignments = state.match_cols.size
    return float(y_rc_mean - state.matched_y_sum / n_assignments)


def state_balance_l2(
    state: ConstrainedOneToKState,
    x_rc_mean: np.ndarray,
) -> float:
    """L2 norm of recipient-vs-matched-donor mean covariate imbalance."""
    n_assignments = state.match_cols.size
    matched_mean = state.matched_x_sum / n_assignments
    return float(np.linalg.norm(x_rc_mean - matched_mean))


def state_fraction_initial_retained(state: ConstrainedOneToKState) -> float:
    """Fraction of initial recipient-donor assignments still retained."""
    return float(state.retained_count / state.match_cols.size)


def mh_step_constrained_1tok(
    state: ConstrainedOneToKState,
    dist: np.ndarray,
    X_dn: np.ndarray,
    Y_dn: np.ndarray,
    beta: float,
    pstar: float,
    rng: np.random.Generator,
) -> tuple[bool, str]:
    """
    Run one local MH proposal for constrained 1-to-K matching.

    Proposal types:
    - matched_swap: swap one matched donor from two recipients.
    - unmatched_swap: replace one matched donor with one currently unmatched donor.

    The target is proportional to exp(-beta * D(C)), so a proposed distance
    change delta is accepted with probability min(1, exp(-beta * delta)).
    """
    match_cols = state.match_cols
    n_r, k = match_cols.shape

    flag = int(rng.choice([1, 2], p=[pstar, 1 - pstar]))

    if flag == 1:
        row1 = int(rng.integers(n_r))
        row2 = int(rng.integers(n_r - 1))
        if row2 >= row1:
            row2 += 1

        slot1 = int(rng.integers(k))
        slot2 = int(rng.integers(k))
        old1 = int(match_cols[row1, slot1])
        old2 = int(match_cols[row2, slot2])

        delta = (
            dist[row1, old2]
            + dist[row2, old1]
            - dist[row1, old1]
            - dist[row2, old2]
        )

        if np.log(rng.random()) < min(0.0, -beta * delta):
            before = (
                row_retained_count(match_cols[row1], state.initial_sets[row1])
                + row_retained_count(match_cols[row2], state.initial_sets[row2])
            )

            match_cols[row1, slot1] = old2
            match_cols[row2, slot2] = old1
            state.d_total += float(delta)

            after = (
                row_retained_count(match_cols[row1], state.initial_sets[row1])
                + row_retained_count(match_cols[row2], state.initial_sets[row2])
            )
            state.retained_count += after - before
            return True, "matched_swap"

        return False, "matched_swap"

    if len(state.unmatched) == 0:
        return False, "unmatched_swap_none"

    row = int(rng.integers(n_r))
    slot = int(rng.integers(k))
    unmatched_pos = int(rng.integers(len(state.unmatched)))
    new = int(state.unmatched[unmatched_pos])
    old = int(match_cols[row, slot])

    delta = dist[row, new] - dist[row, old]

    if np.log(rng.random()) < min(0.0, -beta * delta):
        before = row_retained_count(match_cols[row], state.initial_sets[row])

        match_cols[row, slot] = new
        state.unmatched[unmatched_pos] = old
        state.d_total += float(delta)
        state.matched_y_sum += float(Y_dn[new] - Y_dn[old])
        state.matched_x_sum += X_dn[new] - X_dn[old]

        after = row_retained_count(match_cols[row], state.initial_sets[row])
        state.retained_count += after - before
        return True, "unmatched_swap"

    return False, "unmatched_swap"


def mh_step_constrained_1tok_block(
    state: ConstrainedOneToKState,
    dist: np.ndarray,
    X_dn: np.ndarray,
    Y_dn: np.ndarray,
    beta: float,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[bool, str]:
    """Run one pure mixed-pool block MH proposal.

    The proposal selects ``block_size`` distinct recipients, one matched slot
    from each selected recipient, and the same number of currently unmatched
    donors. It uniformly permutes those ``2 * block_size`` donors, places the
    first half in the selected matched slots, and returns the other half to the
    selected unmatched positions.

    Selecting the same positions and applying the reverse permutation has the
    same probability, so the proposal is symmetric. The target is proportional
    to ``exp(-beta * D(C))`` and no Hastings correction is required.

    This function is a pure block kernel: it does not mix in either proposal
    from :func:`mh_step_constrained_1tok`.

    Returns:
        changed_and_accepted, move_type

    A permutation that leaves every selected matched slot unchanged is treated
    as a self-transition. Reordering only the internal unmatched array does not
    constitute a change to the matching state.
    """
    match_cols = state.match_cols
    n_r, k = match_cols.shape
    n_unmatched = len(state.unmatched)

    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if block_size > n_r:
        raise ValueError(
            f"block_size={block_size} exceeds the number of recipients ({n_r})."
        )
    if block_size > n_unmatched:
        raise ValueError(
            f"block_size={block_size} exceeds the unmatched-pool size "
            f"({n_unmatched})."
        )

    rows = rng.choice(n_r, size=block_size, replace=False)
    slots = rng.integers(k, size=block_size)
    unmatched_pos = rng.choice(n_unmatched, size=block_size, replace=False)

    old_matched = match_cols[rows, slots].copy()
    old_unmatched = state.unmatched[unmatched_pos].copy()
    pool = np.concatenate((old_matched, old_unmatched))
    proposed_pool = rng.permutation(pool)
    new_matched = proposed_pool[:block_size]

    if np.array_equal(new_matched, old_matched):
        return False, f"block_pool_b{block_size}_stay"

    delta = float(np.sum(dist[rows, new_matched] - dist[rows, old_matched]))
    if np.log(rng.random()) >= min(0.0, -beta * delta):
        return False, f"block_pool_b{block_size}"

    before = sum(
        row_retained_count(match_cols[int(row)], state.initial_sets[int(row)])
        for row in rows
    )

    match_cols[rows, slots] = new_matched
    state.unmatched[unmatched_pos] = proposed_pool[block_size:]
    state.d_total += delta
    state.matched_y_sum += float(
        np.sum(Y_dn[new_matched]) - np.sum(Y_dn[old_matched])
    )
    state.matched_x_sum += (
        X_dn[new_matched].sum(axis=0) - X_dn[old_matched].sum(axis=0)
    )

    after = sum(
        row_retained_count(match_cols[int(row)], state.initial_sets[int(row)])
        for row in rows
    )
    state.retained_count += after - before
    return True, f"block_pool_b{block_size}"


def gibbs_step_constrained_1tok(
    state: ConstrainedOneToKState,
    dist: np.ndarray,
    X_dn: np.ndarray,
    Y_dn: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> tuple[bool, str]:
    """
    Run one local Gibbs update for constrained 1-to-K matching.

    One local Gibbs update chooses one recipient row and one of its K donor
    slots. Conditional on all other assignments, that slot may keep its current
    donor or switch to one currently unmatched donor.

    Candidate set:
        {current donor in the chosen slot} union {all unmatched donors}

    Conditional weights:
        weight(candidate j) proportional to exp(-beta * dist[row, j])

    Important limitation:
    - If rho = 1, there are no unmatched donors. The candidate set contains
      only the current donor, so this update is degenerate and cannot move.
    - If rho is only slightly above 1, movement may still be slow because the
      available unmatched pool is small.

    Returns:
        changed, move_type

    For Gibbs, "changed" means the sampled candidate was not the current donor.
    There is no accept/reject step.
    """
    match_cols = state.match_cols
    n_r, k = match_cols.shape

    if len(state.unmatched) == 0:
        return False, "gibbs_degenerate_no_unmatched"

    row = int(rng.integers(n_r))
    slot = int(rng.integers(k))
    old = int(match_cols[row, slot])

    # Slot 0 is the current donor; slots 1: are currently unmatched donors.
    candidates = np.empty(len(state.unmatched) + 1, dtype=int)
    candidates[0] = old
    candidates[1:] = state.unmatched

    # Shift the distances for numerical stability. This leaves probabilities
    # unchanged because all weights in this conditional get the same multiplier.
    candidate_dist = dist[row, candidates]
    shifted = candidate_dist - candidate_dist.min()
    weights = np.exp(-beta * shifted)
    cdf = np.cumsum(weights)
    selected_pos = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))

    if selected_pos == 0:
        return False, "gibbs_stay"

    unmatched_pos = selected_pos - 1
    new = int(state.unmatched[unmatched_pos])

    before = row_retained_count(match_cols[row], state.initial_sets[row])

    match_cols[row, slot] = new
    state.unmatched[unmatched_pos] = old
    state.d_total += float(dist[row, new] - dist[row, old])
    state.matched_y_sum += float(Y_dn[new] - Y_dn[old])
    state.matched_x_sum += X_dn[new] - X_dn[old]

    after = row_retained_count(match_cols[row], state.initial_sets[row])
    state.retained_count += after - before
    return True, "gibbs_change"
