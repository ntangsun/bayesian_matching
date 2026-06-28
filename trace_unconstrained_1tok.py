#!/usr/bin/env python3
"""
python trace_bias_by_replication.py `
    --datasets datasets_sb01_10000.npz `
    --n-mcmc 500 `
    --print-every 100 `
    --flush-every 100 `
    --out-csv results\unconstrained_1tok_bias_by_replication_10000.csv `
    --out-plot results\unconstrained_1tok_bias_by_replication_10000.png
    
trace_unconstrained_1tok.py

Track how the estimated treatment effect evolves as a function of the number
of MCMC imputations for unconstrained 1-to-k Bayesian matching.

For each simulation replication, this script runs the full MCMC chain
sequentially, keeps an online running mean of tau, and saves checkpoint
snapshots. After all replications finish, it averages matching checkpoints
across replications and makes a convergence plot.

Example:
    python trace_unconstrained_1tok.py --datasets datasets_sb01.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_matching_from_datasets import (
    distance_matrix,
    lm_slope_variance,
    recipient_donor_indices,
)


def setup_unconstrained_1tok(X, Y, g, beta, k):
    """
    Build the fixed objects needed by the unconstrained 1-to-k sampler.

    donor_order[r, :k] are selected donors for recipient r.
    donor_order[r, k:] are currently unselected donors for recipient r.
    """
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)
    n_dn = len(dn)

    if not 1 <= k < n_dn:
        raise ValueError("k must be at least 1 and smaller than the number of donors.")

    dist = distance_matrix(X, g, 2)
    offset = dist.min(axis=1)
    dist_df = dist + offset[:, None]

    # The original R code uses p but does not define it; keep the project
    # convention that p equals the tempering parameter beta.
    alpha = 1 / (dist_df ** beta)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    donor_order = np.argsort(dist, axis=1)
    Y_rc = Y[rc]
    Y_matches = np.empty((n_rc, k))
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(k * n_rc)])

    return rc, dn, alpha, donor_order, Y_rc, Y_matches, g_oc


def run_one_trace(X, Y, g, seed, sim_number, n_mcmc, checkpoint_every, beta, k):
    """
    Run one replication and return checkpoint rows.

    tau_sum is updated once per MCMC iteration, so checkpoint running means are
    computed online without storing or rescanning the full tau history.
    """
    rng = np.random.default_rng(int(seed) + 1_000_000)
    rc, dn, alpha, donor_order, Y_rc, Y_matches, g_oc = setup_unconstrained_1tok(
        X, Y, g, beta, k
    )

    n_rc = len(rc)
    n_dn = len(dn)
    tau_sum = 0.0
    accepted_iterations = 0
    accepted_row_updates = 0
    rows = []

    for mcmc_iter in range(1, n_mcmc + 1):
        accepted_any = False

        # -----------------------------
        # MCMC Sampling
        # -----------------------------
        for r in range(n_rc):
            # Sample one position from the selected block and one from the
            # unselected block. If accepted, swap those entries in place.
            remove_pos = rng.integers(k)
            add_pos = rng.integers(k, n_dn)
            x1 = donor_order[r, remove_pos]
            x0 = donor_order[r, add_pos]

            prob_x1 = alpha[r, x1]
            prob_x0 = alpha[r, x0]
            w0 = prob_x0 / (1 - prob_x0)
            w1 = prob_x1 / (1 - prob_x1)

            logr = min(0.0, np.log(w0 / w1))
            if np.log(rng.random()) < logr:
                donor_order[r, remove_pos] = x0
                donor_order[r, add_pos] = x1
                accepted_any = True
                accepted_row_updates += 1

            Y_matches[r, :] = Y[dn[donor_order[r, :k]]]

        if accepted_any:
            accepted_iterations += 1

        # -----------------------------
        # Online Tau Update
        # -----------------------------
        Y_match = Y_matches.ravel()
        Y_oc = np.concatenate([Y_rc, Y_match])
        tau_hat, _ = lm_slope_variance(Y_oc, g_oc)

        tau_sum += tau_hat
        running_tau = tau_sum / mcmc_iter

        # -----------------------------
        # Checkpoint Snapshot
        # -----------------------------
        if mcmc_iter % checkpoint_every == 0 or mcmc_iter == n_mcmc:
            rows.append(
                {
                    "sim": sim_number,
                    "mcmc_iter": mcmc_iter,
                    "tau_current": tau_hat,
                    "running_tau": running_tau,
                    "accepted_iterations": accepted_iterations,
                    "accepted_iteration_rate": accepted_iterations / mcmc_iter,
                    "accepted_row_updates": accepted_row_updates,
                    "accepted_row_update_rate": accepted_row_updates / (mcmc_iter * n_rc),
                }
            )

    return rows


def summarize_trace(detail_df, true_tau):
    """Average matching checkpoints across simulation replications."""
    summary = (
        detail_df.groupby("mcmc_iter", as_index=False)
        .agg(
            n_sim=("sim", "nunique"),
            mean_tau=("running_tau", "mean"),
            sd_tau=("running_tau", "std"),
            q05_tau=("running_tau", lambda x: x.quantile(0.05)),
            q95_tau=("running_tau", lambda x: x.quantile(0.95)),
            mean_acceptance_rate=("accepted_iteration_rate", "mean"),
            mean_row_update_rate=("accepted_row_update_rate", "mean"),
        )
    )
    summary["bias"] = summary["mean_tau"] - true_tau
    summary["abs_bias"] = summary["bias"].abs()
    summary["se_tau"] = summary["sd_tau"] / np.sqrt(summary["n_sim"])
    return summary


def plot_trace(summary_df, true_tau, out_plot):
    """Save a two-panel plot for mean tau and bias across checkpoints."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    x = summary_df["mcmc_iter"].to_numpy()
    mean_tau = summary_df["mean_tau"].to_numpy()
    q05 = summary_df["q05_tau"].to_numpy()
    q95 = summary_df["q95_tau"].to_numpy()
    bias = summary_df["bias"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(x, mean_tau, color="#1f77b4", linewidth=2, label="Mean running tau")
    axes[0].fill_between(x, q05, q95, color="#1f77b4", alpha=0.18, label="5th-95th percentile")
    axes[0].axhline(true_tau, color="black", linestyle="--", linewidth=1, label="True tau")
    axes[0].set_ylabel("Estimated tau")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, bias, color="#d62728", linewidth=2, label="Bias")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of MCMC imputations")
    axes[1].set_ylabel("Bias")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Unconstrained 1-to-k Tau Trace Across Simulation Replications")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets_sb01.npz")
    parser.add_argument("--n-mcmc", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--true-tau", type=float, default=1.0)
    parser.add_argument("--max-sim", type=int, default=None, help="Optional smoke-test limit; default uses all replications")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--out-detail", type=str, default="results/unconstrained_1tok_tau_trace_by_sim.csv")
    parser.add_argument("--out-summary", type=str, default="results/unconstrained_1tok_tau_trace_summary.csv")
    parser.add_argument("--out-plot", type=str, default="results/unconstrained_1tok_tau_trace.png")
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive.")

    data = np.load(args.datasets)
    X_all = data["X"]
    Y_all = data["Y"]
    g_all = data["g"]
    seeds = data["seeds"] if "seeds" in data else np.arange(1, len(X_all) + 1)

    n_sim = len(X_all)
    if args.max_sim is not None:
        n_sim = min(n_sim, args.max_sim)

    all_rows = []
    for s in range(n_sim):
        if s == 0 or (s + 1) % args.print_every == 0 or (s + 1) == n_sim:
            print(f"Trace simulation {s + 1}/{n_sim}")

        rows = run_one_trace(
            X_all[s],
            Y_all[s],
            g_all[s],
            seeds[s],
            sim_number=s + 1,
            n_mcmc=args.n_mcmc,
            checkpoint_every=args.checkpoint_every,
            beta=args.beta,
            k=args.k,
        )
        all_rows.extend(rows)

    detail_df = pd.DataFrame(all_rows)
    summary_df = summarize_trace(detail_df, true_tau=args.true_tau)

    out_detail = Path(args.out_detail)
    out_summary = Path(args.out_summary)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    detail_df.to_csv(out_detail, index=False)
    summary_df.to_csv(out_summary, index=False)
    plot_trace(summary_df, true_tau=args.true_tau, out_plot=args.out_plot)

    print(f"Saved detail trace to {out_detail}")
    print(f"Saved summary trace to {out_summary}")
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
