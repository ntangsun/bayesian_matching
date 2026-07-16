#!/usr/bin/env python3
r"""
python experiments/traces/trace_constrained_1tok.py `
    --datasets datasets_sb01.npz `
    --sim-index 1 `
    --n-mcmc 10000 `
    --checkpoint-every 100 `
    --k 2 `
    --beta 1 `
    --pstar 0.5 `
    --out-csv results\experiments\fixed_rep_traces\constrained_1tok\constrained_1tok_fixed_rep_trace.csv `
    --out-plot results\experiments\fixed_rep_traces\constrained_1tok\constrained_1tok_fixed_rep_trace.png

experiments/traces/trace_constrained_1tok.py

Trace one constrained 1-to-k Bayesian matching Markov chain on one fixed
dataset replication.

This is a chain-convergence diagnostic, not a Monte Carlo simulation summary
across many generated datasets. It keeps the dataset fixed, runs the same
constrained 1-to-k proposal logic as run_matching_from_datasets.py, and records
checkpoint snapshots of the running tau estimate.

Example:
    python experiments/traces/trace_constrained_1tok.py --datasets datasets_sb01.npz --sim-index 1 --k 2
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_matching_from_datasets import (
    distance_matrix,
    init_constrained_greedy,
    recipient_donor_indices,
    wls_slope_variance,
)


def setup_constrained_1tok(X, Y, g, k):
    """Build fixed objects and a feasible initial constrained 1-to-k match."""
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)
    n_dn = len(dn)

    if k < 1 or n_rc * k > n_dn:
        raise ValueError("constrained 1-to-k matching requires k >= 1 and n_rc * k <= n_dn.")

    dist = distance_matrix(X, g, 2)
    C_acc = init_constrained_greedy(dist, k=k)
    match_cols = np.array([np.where(C_acc[s] == 1)[0] for s in range(n_rc)])
    unmatched_donors = np.where(C_acc.sum(axis=0) == 0)[0]

    Y_rc = Y[rc]
    Y_matches = np.empty((n_rc, k))
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(k * n_rc)])
    weights = np.concatenate([np.ones(n_rc), np.full(k * n_rc, 1 / k)])

    return rc, dn, dist, match_cols, unmatched_donors, Y_rc, Y_matches, g_oc, weights


def matched_outputs(Y, dn, match_cols, Y_matches):
    """Return current donor outcome matches for the constrained 1-to-k state."""
    Y_matches[:, :] = Y[dn[match_cols]]
    return Y_matches.ravel()


def total_match_distance(dist, match_cols):
    """Compute the total current distance over all selected recipient-donor pairs."""
    rows = np.arange(match_cols.shape[0])[:, None]
    return float(dist[rows, match_cols].sum())


def propose_constrained_1tok(match_cols, unmatched_donors, dist, beta, pstar, rng):
    """
    Propose one constrained 1-to-k MCMC update.

    Type 1 swaps one matched donor slot between two recipients. Type 2 replaces
    one matched donor slot with one currently unmatched donor.
    """
    n_rc, k = match_cols.shape
    flag = rng.choice([1, 2], p=[pstar, 1 - pstar])

    if flag == 1:
        row1 = rng.integers(n_rc)
        row2 = rng.integers(n_rc - 1)
        if row2 >= row1:
            row2 += 1

        slot1 = rng.integers(k)
        slot2 = rng.integers(k)
        old1 = match_cols[row1, slot1]
        old2 = match_cols[row2, slot2]

        loglik_nr = -beta * (dist[row1, old2] + dist[row2, old1])
        loglik_dr = -beta * (dist[row1, old1] + dist[row2, old2])

    else:
        if len(unmatched_donors) == 0:
            loglik_nr = -np.inf
            loglik_dr = 0.0
        else:
            unmatched_pos = rng.integers(len(unmatched_donors))
            dn_new = unmatched_donors[unmatched_pos]
            rc_new = rng.integers(0, n_rc)
            slot = rng.integers(k)
            old = match_cols[rc_new, slot]

            loglik_nr = -beta * dist[rc_new, dn_new]
            loglik_dr = -beta * dist[rc_new, old]

    logr = min(0.0, loglik_nr - loglik_dr)
    accepted = np.log(rng.random()) < logr

    if accepted:
        if flag == 1:
            match_cols[row1, slot1] = old2
            match_cols[row2, slot2] = old1
        else:
            if len(unmatched_donors) > 0:
                match_cols[rc_new, slot] = dn_new
                unmatched_donors[unmatched_pos] = old

    return accepted, int(flag)


def run_trace(X, Y, g, seed, sim_index, n_mcmc, checkpoint_every, k, beta, pstar):
    """Run one fixed-dataset constrained 1-to-k chain and return snapshots."""
    rng = np.random.default_rng(int(seed) + 1_000_000)
    (
        _,
        dn,
        dist,
        match_cols,
        unmatched_donors,
        Y_rc,
        Y_matches,
        g_oc,
        weights,
    ) = setup_constrained_1tok(X, Y, g, k)

    tau_sum = 0.0
    accepted_iterations = 0
    type1_count = 0
    type2_count = 0
    type1_accepts = 0
    type2_accepts = 0
    rows = []

    for mcmc_iter in range(1, n_mcmc + 1):
        accepted, update_type = propose_constrained_1tok(
            match_cols, unmatched_donors, dist, beta, pstar, rng
        )

        if update_type == 1:
            type1_count += 1
            if accepted:
                type1_accepts += 1
        else:
            type2_count += 1
            if accepted:
                type2_accepts += 1

        if accepted:
            accepted_iterations += 1

        Y_match = matched_outputs(Y, dn, match_cols, Y_matches)
        Y_oc = np.concatenate([Y_rc, Y_match])
        tau_hat, _ = wls_slope_variance(Y_oc, g_oc, weights)
        tau_sum += tau_hat
        running_tau = tau_sum / mcmc_iter

        if mcmc_iter % checkpoint_every == 0 or mcmc_iter == n_mcmc:
            rows.append(
                {
                    "sim": sim_index,
                    "mcmc_iter": mcmc_iter,
                    "k": k,
                    "tau_current": tau_hat,
                    "running_tau": running_tau,
                    "total_distance": total_match_distance(dist, match_cols),
                    "accepted_iterations": accepted_iterations,
                    "accepted_iteration_rate": accepted_iterations / mcmc_iter,
                    "type1_count": type1_count,
                    "type2_count": type2_count,
                    "type1_acceptance_rate": type1_accepts / type1_count if type1_count else np.nan,
                    "type2_acceptance_rate": type2_accepts / type2_count if type2_count else np.nan,
                }
            )

    return rows


def plot_trace(trace_df, true_tau, out_plot):
    """Save a compact chain diagnostic plot."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    x = trace_df["mcmc_iter"].to_numpy()
    k = int(trace_df["k"].iloc[0]) if "k" in trace_df else 2

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, trace_df["tau_current"], color="#9ecae1", linewidth=1, label="Current tau")
    axes[0].plot(x, trace_df["running_tau"], color="#1f77b4", linewidth=2, label="Running tau")
    axes[0].axhline(true_tau, color="black", linestyle="--", linewidth=1, label="True tau")
    axes[0].set_ylabel("Tau")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, trace_df["total_distance"], color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("Total distance")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, trace_df["accepted_iteration_rate"], color="#d62728", linewidth=2)
    axes[2].set_xlabel("MCMC iteration")
    axes[2].set_ylabel("Acceptance rate")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"Constrained 1-to-{k} MCMC Trace on One Fixed Dataset")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets_sb01.npz")
    parser.add_argument("--sim-index", type=int, default=1, help="One-based dataset replication to trace")
    parser.add_argument("--n-mcmc", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--pstar", type=float, default=0.5)
    parser.add_argument("--true-tau", type=float, default=1.0)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/experiments/fixed_rep_traces/constrained_1tok/constrained_1tok_fixed_rep_trace.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="results/experiments/fixed_rep_traces/constrained_1tok/constrained_1tok_fixed_rep_trace.png",
    )
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive.")
    if not 0 <= args.pstar <= 1:
        raise ValueError("pstar must be between 0 and 1.")

    data = np.load(args.datasets)
    X_all = data["X"]
    Y_all = data["Y"]
    g_all = data["g"]
    seeds = data["seeds"] if "seeds" in data else np.arange(1, len(X_all) + 1)

    if not 1 <= args.sim_index <= len(X_all):
        raise ValueError("sim-index must be between 1 and the number of datasets.")

    s = args.sim_index - 1
    print(
        f"Tracing constrained 1-to-{args.k} chain on dataset {args.sim_index} "
        f"for {args.n_mcmc} iterations"
    )

    rows = run_trace(
        X_all[s],
        Y_all[s],
        g_all[s],
        seeds[s],
        sim_index=args.sim_index,
        n_mcmc=args.n_mcmc,
        checkpoint_every=args.checkpoint_every,
        k=args.k,
        beta=args.beta,
        pstar=args.pstar,
    )

    trace_df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(out_csv, index=False)
    plot_trace(trace_df, true_tau=args.true_tau, out_plot=args.out_plot)

    print(f"Saved trace CSV to {out_csv}")
    print(f"Saved trace plot to {args.out_plot}")


if __name__ == "__main__":
    main()
