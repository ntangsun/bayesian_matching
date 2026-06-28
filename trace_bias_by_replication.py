#!/usr/bin/env python3
"""
trace_bias_by_replication.py

Estimate how simulation bias stabilizes as the number of replications grows.

For each generated dataset, this script runs unconstrained 1-to-k matching for a
fixed number of MCMC imputations, averages tau within that replication, and then
updates the running average across replications. The plot answers:

    How many simulation replications do we need before estimated bias stabilizes?

Example:
    python trace_bias_by_replication.py --datasets datasets_sb01_10000.npz
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_matching_from_datasets import (
    distance_matrix,
    lm_slope_variance,
    recipient_donor_indices,
)


def setup_unconstrained_1tok(X, Y, g, beta, k):
    """
    Build fixed objects for one unconstrained 1-to-k replication.

    donor_order[r, :k] are currently matched donors for recipient r.
    donor_order[r, k:] are currently unmatched donors for recipient r.
    """
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)
    n_dn = len(dn)

    if not 1 <= k < n_dn:
        raise ValueError("k must be at least 1 and smaller than the number of donors.")

    dist = distance_matrix(X, g, 2)
    offset = dist.min(axis=1)
    dist_df = dist + offset[:, None]

    # Match the convention used elsewhere in this project: p = beta.
    alpha = 1 / (dist_df ** beta)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    donor_order = np.argsort(dist, axis=1)
    Y_rc = Y[rc]
    Y_matches = np.empty((n_rc, k))
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(k * n_rc)])

    return rc, dn, alpha, donor_order, Y_rc, Y_matches, g_oc


def estimate_tau_one_replication(X, Y, g, seed, n_mcmc, beta, k):
    """
    Run one replication and return the MCMC-averaged tau estimate.

    tau_sum is updated online, so we do not store the 500 tau estimates.
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

    for _ in range(n_mcmc):
        accepted_any = False

        # -----------------------------
        # MCMC Sampling
        # -----------------------------
        for r in range(n_rc):
            # Sample one selected donor and one unselected donor by position.
            # Accepted proposals are in-place swaps in donor_order.
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

    return {
        "tau_hat": tau_sum / n_mcmc,
        "accepted_iteration_rate": accepted_iterations / n_mcmc,
        "accepted_row_update_rate": accepted_row_updates / (n_mcmc * n_rc),
    }


def plot_running_bias(rows, true_tau, out_plot):
    """Save the running bias plot."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    sims = np.array([row["sim"] for row in rows])
    running_bias = np.array([row["running_bias"] for row in rows])
    running_mean_tau = np.array([row["running_mean_tau"] for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(sims, running_mean_tau, color="#1f77b4", linewidth=1.8)
    axes[0].axhline(true_tau, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Running mean tau")
    axes[0].grid(alpha=0.25)

    axes[1].plot(sims, running_bias, color="#d62728", linewidth=1.8)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of simulation replications")
    axes[1].set_ylabel("Running bias")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Unconstrained 1-to-k Bias Stability Across Replications")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets_sb01_10000.npz")
    parser.add_argument("--n-mcmc", type=int, default=500)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--true-tau", type=float, default=1.0)
    parser.add_argument("--max-sim", type=int, default=None, help="Optional smoke-test limit; default uses all replications")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--out-csv", type=str, default="results/unconstrained_1tok_bias_by_replication.csv")
    parser.add_argument("--out-plot", type=str, default="results/unconstrained_1tok_bias_by_replication.png")
    args = parser.parse_args()

    data = np.load(args.datasets)
    X_all = data["X"]
    Y_all = data["Y"]
    g_all = data["g"]
    seeds = data["seeds"] if "seeds" in data else np.arange(1, len(X_all) + 1)

    n_sim = len(X_all)
    if args.max_sim is not None:
        n_sim = min(n_sim, args.max_sim)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sim",
        "tau_hat",
        "running_mean_tau",
        "running_bias",
        "running_abs_bias",
        "accepted_iteration_rate",
        "accepted_row_update_rate",
        "n_mcmc",
        "k",
        "beta",
        "dataset_file",
    ]

    rows = []
    tau_sum_across_reps = 0.0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in range(n_sim):
            if s == 0 or (s + 1) % args.print_every == 0 or (s + 1) == n_sim:
                print(f"Bias trace simulation {s + 1}/{n_sim}")

            result = estimate_tau_one_replication(
                X_all[s],
                Y_all[s],
                g_all[s],
                seeds[s],
                n_mcmc=args.n_mcmc,
                beta=args.beta,
                k=args.k,
            )

            tau_sum_across_reps += result["tau_hat"]
            running_mean_tau = tau_sum_across_reps / (s + 1)
            running_bias = running_mean_tau - args.true_tau

            row = {
                "sim": s + 1,
                "tau_hat": result["tau_hat"],
                "running_mean_tau": running_mean_tau,
                "running_bias": running_bias,
                "running_abs_bias": abs(running_bias),
                "accepted_iteration_rate": result["accepted_iteration_rate"],
                "accepted_row_update_rate": result["accepted_row_update_rate"],
                "n_mcmc": args.n_mcmc,
                "k": args.k,
                "beta": args.beta,
                "dataset_file": str(args.datasets),
            }
            rows.append(row)
            writer.writerow(row)

            if (s + 1) % args.flush_every == 0:
                f.flush()

    plot_running_bias(rows, true_tau=args.true_tau, out_plot=args.out_plot)
    print(f"Saved running bias CSV to {out_csv}")
    print(f"Saved running bias plot to {args.out_plot}")


if __name__ == "__main__":
    main()
