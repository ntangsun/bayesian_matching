#!/usr/bin/env python3
"""
python trace_constrained_1to1.py `
    --datasets datasets_sb01.npz `
    --sim-index 1 `
    --n-mcmc 10000 `
    --checkpoint-every 100 `
    --beta 1 `
    --pstar 0.5 `
    --out-csv results\constrained_1to1_fixed_rep_trace.csv `
    --out-plot results\constrained_1to1_fixed_rep_trace.png

trace_constrained_1to1.py

Trace one constrained 1-to-1 Bayesian matching Markov chain on one fixed
dataset replication.

This is a chain-convergence diagnostic, not a Monte Carlo simulation summary
across many generated datasets. It keeps the dataset fixed, runs the same
constrained 1-to-1 proposal logic as run_matching_from_datasets.py, and records
checkpoint snapshots of the running tau estimate.

Example:
    python trace_constrained_1to1.py --datasets datasets_sb01.npz --sim-index 1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_matching_from_datasets import (
    distance_matrix,
    init_constrained_greedy,
    lm_slope_variance,
    recipient_donor_indices,
)


def setup_constrained_1to1(X, Y, g):
    """Build fixed objects and a feasible initial constrained 1-to-1 match."""
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)
    n_dn = len(dn)

    if n_dn < n_rc:
        raise ValueError("constrained 1-to-1 matching requires at least as many donors as recipients.")

    dist = distance_matrix(X, g, 2)
    C_acc = init_constrained_greedy(dist, k=1)
    Y_rc = Y[rc]
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(n_rc)])

    return rc, dn, dist, C_acc, Y_rc, g_oc


def matched_outputs(Y, dn, C_acc):
    """Return current donor outcome matches for the constrained match matrix."""
    match_cols = np.argmax(C_acc, axis=1)
    return match_cols, Y[dn[match_cols]]


def total_match_distance(dist, C_acc):
    """Compute the total current distance over selected recipient-donor pairs."""
    match_cols = np.argmax(C_acc, axis=1)
    return float(dist[np.arange(len(match_cols)), match_cols].sum())


def propose_constrained_1to1(C_acc, dist, beta, pstar, rng):
    """
    Propose one constrained 1-to-1 MCMC update.

    Type 1 swaps donor assignments between two recipients. Type 2 replaces one
    recipient's current donor with one currently unmatched donor.
    """
    n_rc, _ = C_acc.shape
    flag = rng.choice([1, 2], p=[pstar, 1 - pstar])

    collist = np.where(C_acc.sum(axis=0) == 0)[0]
    C_pr = C_acc.copy()

    if flag == 1:
        rows = rng.choice(n_rc, size=2, replace=False)

        C_pr[rows[0], :] = C_acc[rows[1], :]
        C_pr[rows[1], :] = C_acc[rows[0], :]

        new1 = np.where(C_pr[rows[0]] == 1)[0][0]
        new2 = np.where(C_pr[rows[1]] == 1)[0][0]
        old1 = np.where(C_acc[rows[0]] == 1)[0][0]
        old2 = np.where(C_acc[rows[1]] == 1)[0][0]

        loglik_nr = -beta * (dist[rows[0], new1] + dist[rows[1], new2])
        loglik_dr = -beta * (dist[rows[0], old1] + dist[rows[1], old2])

    else:
        if len(collist) == 0:
            loglik_nr = -np.inf
            loglik_dr = 0.0
        else:
            dn_new = rng.choice(collist)
            rc_new = rng.integers(0, n_rc)

            old = np.where(C_acc[rc_new] == 1)[0][0]
            C_pr[rc_new, :] = 0
            C_pr[rc_new, dn_new] = 1

            loglik_nr = -beta * dist[rc_new, dn_new]
            loglik_dr = -beta * dist[rc_new, old]

    loglik_ratio = loglik_nr - loglik_dr
    logr = min(0.0, loglik_ratio)
    accepted = np.log(rng.random()) < logr

    if accepted:
        return C_pr, True, int(flag)
    return C_acc, False, int(flag)


def run_trace(X, Y, g, seed, sim_index, n_mcmc, checkpoint_every, beta, pstar):
    """Run one fixed-dataset constrained 1-to-1 chain and return snapshots."""
    rng = np.random.default_rng(int(seed) + 1_000_000)
    rc, dn, dist, C_acc, Y_rc, g_oc = setup_constrained_1to1(X, Y, g)

    tau_sum = 0.0
    accepted_iterations = 0
    type1_count = 0
    type2_count = 0
    type1_accepts = 0
    type2_accepts = 0
    rows = []

    for mcmc_iter in range(1, n_mcmc + 1):
        C_acc, accepted, update_type = propose_constrained_1to1(
            C_acc, dist, beta, pstar, rng
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

        _, Y_match = matched_outputs(Y, dn, C_acc)
        Y_oc = np.concatenate([Y_rc, Y_match])
        tau_hat, _ = lm_slope_variance(Y_oc, g_oc)
        tau_sum += tau_hat
        running_tau = tau_sum / mcmc_iter

        if mcmc_iter % checkpoint_every == 0 or mcmc_iter == n_mcmc:
            rows.append(
                {
                    "sim": sim_index,
                    "mcmc_iter": mcmc_iter,
                    "tau_current": tau_hat,
                    "running_tau": running_tau,
                    "total_distance": total_match_distance(dist, C_acc),
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

    fig.suptitle("Constrained 1-to-1 MCMC Trace on One Fixed Dataset")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets_sb01.npz")
    parser.add_argument("--sim-index", type=int, default=1, help="One-based dataset replication to trace")
    parser.add_argument("--n-mcmc", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--pstar", type=float, default=0.5)
    parser.add_argument("--true-tau", type=float, default=1.0)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/constrained_1to1_fixed_rep_trace.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="results/constrained_1to1_fixed_rep_trace.png",
    )
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive.")

    data = np.load(args.datasets)
    X_all = data["X"]
    Y_all = data["Y"]
    g_all = data["g"]
    seeds = data["seeds"] if "seeds" in data else np.arange(1, len(X_all) + 1)

    if not 1 <= args.sim_index <= len(X_all):
        raise ValueError("sim-index must be between 1 and the number of datasets.")

    s = args.sim_index - 1
    print(
        f"Tracing constrained 1-to-1 chain on dataset {args.sim_index} "
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
