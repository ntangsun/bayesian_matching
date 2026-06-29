#!/usr/bin/env python3
r"""
python trace_compare_constrained_1to1.py `
    --datasets datasets_sb01.npz `
    --sim-index 1 `
    --n-mcmc 10000 `
    --checkpoint-every 100 `
    --beta 1 `
    --pstar 0.5 `
    --out-detail results\constrained_1to1_mcmc_vs_gibbs_detail.csv `
    --out-comparison results\constrained_1to1_mcmc_vs_gibbs_comparison.csv `
    --out-plot results\constrained_1to1_mcmc_vs_gibbs.png

trace_compare_constrained_1to1.py

Compare the Metropolis constrained 1-to-1 sampler with the Gibbs constrained
1-to-1 sampler on one fixed dataset replication.

The diagnostic is estimator-focused: both chains run on the same fixed data,
record the running mean of the same tau estimator, and save checkpointed
differences. This checks whether the two samplers appear to target the same
matching distribution for that dataset.

Example:
    python trace_compare_constrained_1to1.py `
        --datasets datasets_sb01.npz `
        --sim-index 1 `
        --n-mcmc 10000 `
        --checkpoint-every 100 `
        --out-comparison results\constrained_1to1_mcmc_vs_gibbs_comparison.csv `
        --out-plot results\constrained_1to1_mcmc_vs_gibbs.png
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


def setup_common(X, Y, g, beta):
    """Build common fixed data and initial constrained 1-to-1 state."""
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)
    n_dn = len(dn)

    if n_dn < n_rc:
        raise ValueError("constrained 1-to-1 matching requires n_dn >= n_rc.")

    dist = distance_matrix(X, g, 2)
    C_init = init_constrained_greedy(dist, k=1)
    match_cols = np.argmax(C_init, axis=1)
    unmatched_donors = np.where(C_init.sum(axis=0) == 0)[0]

    shifted_dist = dist - dist.min(axis=1, keepdims=True)
    gibbs_weights = np.exp(-beta * shifted_dist)

    Y_rc = Y[rc]
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(n_rc)])

    return rc, dn, dist, gibbs_weights, match_cols, unmatched_donors, Y_rc, g_oc


def tau_from_match_cols(Y, dn, Y_rc, g_oc, match_cols):
    """Compute tau from the current recipient-to-donor matching."""
    Y_match = Y[dn[match_cols]]
    Y_oc = np.concatenate([Y_rc, Y_match])
    tau_hat, _ = lm_slope_variance(Y_oc, g_oc)
    return tau_hat


def total_match_distance(dist, match_cols):
    """Compute total selected-pair distance for the current matching."""
    return float(dist[np.arange(len(match_cols)), match_cols].sum())


def metropolis_step(match_cols, unmatched_donors, dist, beta, pstar, rng):
    """Advance the Metropolis constrained 1-to-1 sampler by one proposal."""
    n_rc = len(match_cols)
    flag = rng.choice([1, 2], p=[pstar, 1 - pstar])

    if flag == 1:
        row1 = rng.integers(n_rc)
        row2 = rng.integers(n_rc - 1)
        if row2 >= row1:
            row2 += 1

        old1 = match_cols[row1]
        old2 = match_cols[row2]

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
            old = match_cols[rc_new]

            loglik_nr = -beta * dist[rc_new, dn_new]
            loglik_dr = -beta * dist[rc_new, old]

    logr = min(0.0, loglik_nr - loglik_dr)
    accepted = np.log(rng.random()) < logr

    if accepted:
        if flag == 1:
            match_cols[row1] = old2
            match_cols[row2] = old1
        else:
            if len(unmatched_donors) > 0:
                match_cols[rc_new] = dn_new
                unmatched_donors[unmatched_pos] = old

    return accepted


def gibbs_sweep(match_cols, unmatched_donors, gibbs_weights, weights, cdf, rng):
    """Advance the Gibbs constrained 1-to-1 sampler by one full recipient sweep."""
    changed = 0
    n_rc = len(match_cols)
    n_unmatched = len(unmatched_donors)

    for r in range(n_rc):
        current = match_cols[r]

        # Slot 0 is the current donor. Slots 1: are currently unmatched donors.
        weights[0] = gibbs_weights[r, current]
        if n_unmatched:
            weights[1:] = gibbs_weights[r, unmatched_donors]

        np.cumsum(weights, out=cdf)
        slot = np.searchsorted(cdf, rng.random() * cdf[-1], side="right")

        if slot > 0:
            unmatched_pos = slot - 1
            new = unmatched_donors[unmatched_pos]
            match_cols[r] = new
            unmatched_donors[unmatched_pos] = current
            changed += 1

    return changed


def trace_one_sampler(
    sampler,
    X,
    Y,
    g,
    seed,
    sim_index,
    n_mcmc,
    checkpoint_every,
    beta,
    pstar,
):
    """Run one sampler on one fixed dataset and return checkpoint rows."""
    _, dn, dist, gibbs_weights, match_cols, unmatched_donors, Y_rc, g_oc = setup_common(
        X, Y, g, beta
    )

    rng = np.random.default_rng(int(seed))
    tau_sum = 0.0
    accepted_count = 0
    changed_count = 0
    changed_iterations = 0
    rows = []

    weights = np.empty(len(unmatched_donors) + 1)
    cdf = np.empty(len(unmatched_donors) + 1)

    for mcmc_iter in range(1, n_mcmc + 1):
        if sampler == "mcmc":
            accepted = metropolis_step(match_cols, unmatched_donors, dist, beta, pstar, rng)
            if accepted:
                accepted_count += 1
        elif sampler == "gibbs":
            changed = gibbs_sweep(match_cols, unmatched_donors, gibbs_weights, weights, cdf, rng)
            changed_count += changed
            if changed:
                changed_iterations += 1
        else:
            raise ValueError("Unknown sampler.")

        tau_hat = tau_from_match_cols(Y, dn, Y_rc, g_oc, match_cols)
        tau_sum += tau_hat
        running_tau = tau_sum / mcmc_iter

        if mcmc_iter % checkpoint_every == 0 or mcmc_iter == n_mcmc:
            row = {
                "sim": sim_index,
                "sampler": sampler,
                "mcmc_iter": mcmc_iter,
                "tau_current": tau_hat,
                "running_tau": running_tau,
                "total_distance": total_match_distance(dist, match_cols),
            }
            if sampler == "mcmc":
                row["accepted_count"] = accepted_count
                row["accepted_rate"] = accepted_count / mcmc_iter
            else:
                row["changed_count"] = changed_count
                row["changed_iteration_count"] = changed_iterations
                row["changed_iteration_rate"] = changed_iterations / mcmc_iter
                row["changed_recipient_rate"] = changed_count / (mcmc_iter * len(match_cols))
            rows.append(row)

    return rows


def make_comparison(detail_df):
    """Create paired checkpoint rows with running tau and distance differences."""
    tau_wide = (
        detail_df.pivot_table(
            index=["sim", "mcmc_iter"],
            columns="sampler",
            values="running_tau",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"mcmc": "running_tau_mcmc", "gibbs": "running_tau_gibbs"})
    )
    tau_wide["tau_diff_mcmc_minus_gibbs"] = (
        tau_wide["running_tau_mcmc"] - tau_wide["running_tau_gibbs"]
    )

    dist_wide = (
        detail_df.pivot_table(
            index=["sim", "mcmc_iter"],
            columns="sampler",
            values="total_distance",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"mcmc": "total_distance_mcmc", "gibbs": "total_distance_gibbs"})
    )
    dist_wide["total_distance_diff_mcmc_minus_gibbs"] = (
        dist_wide["total_distance_mcmc"] - dist_wide["total_distance_gibbs"]
    )

    return tau_wide.merge(dist_wide, on=["sim", "mcmc_iter"], how="left")


def plot_comparison(comparison_df, true_tau, out_plot):
    """Plot running tau and MCMC-minus-Gibbs differences."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    x = comparison_df["mcmc_iter"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, comparison_df["running_tau_mcmc"], color="#1f77b4", linewidth=2, label="MCMC")
    axes[0].plot(x, comparison_df["running_tau_gibbs"], color="#2ca02c", linewidth=2, label="Gibbs")
    axes[0].axhline(true_tau, color="black", linestyle="--", linewidth=1, label="True tau")
    axes[0].set_ylabel("Running tau")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, comparison_df["tau_diff_mcmc_minus_gibbs"], color="#d62728", linewidth=2)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Tau diff")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, comparison_df["total_distance_mcmc"], color="#1f77b4", linewidth=2, label="MCMC")
    axes[2].plot(x, comparison_df["total_distance_gibbs"], color="#2ca02c", linewidth=2, label="Gibbs")
    axes[2].set_xlabel("MCMC iteration")
    axes[2].set_ylabel("Total distance")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Constrained 1-to-1: Metropolis vs Gibbs on One Fixed Dataset")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets_sb01.npz")
    parser.add_argument("--sim-index", type=int, default=1, help="One-based dataset replication to trace")
    parser.add_argument("--n-mcmc", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--pstar", type=float, default=0.5)
    parser.add_argument("--true-tau", type=float, default=1.0)
    parser.add_argument(
        "--out-detail",
        type=str,
        default="results/constrained_1to1_mcmc_vs_gibbs_detail.csv",
    )
    parser.add_argument(
        "--out-comparison",
        type=str,
        default="results/constrained_1to1_mcmc_vs_gibbs_comparison.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="results/constrained_1to1_mcmc_vs_gibbs.png",
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
        f"Comparing constrained 1-to-1 MCMC and Gibbs on dataset {args.sim_index} "
        f"for {args.n_mcmc} iterations"
    )

    all_rows = []
    for sampler, seed_offset in [("mcmc", 1_000_000), ("gibbs", 2_000_000)]:
        all_rows.extend(
            trace_one_sampler(
                sampler,
                X_all[s],
                Y_all[s],
                g_all[s],
                int(seeds[s]) + seed_offset,
                sim_index=args.sim_index,
                n_mcmc=args.n_mcmc,
                checkpoint_every=args.checkpoint_every,
                beta=args.beta,
                pstar=args.pstar,
            )
        )

    detail_df = pd.DataFrame(all_rows)
    comparison_df = make_comparison(detail_df)

    out_detail = Path(args.out_detail)
    out_comparison = Path(args.out_comparison)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    out_comparison.parent.mkdir(parents=True, exist_ok=True)

    detail_df.to_csv(out_detail, index=False)
    comparison_df.to_csv(out_comparison, index=False)
    plot_comparison(comparison_df, true_tau=args.true_tau, out_plot=args.out_plot)

    print(f"Saved detail trace to {out_detail}")
    print(f"Saved comparison trace to {out_comparison}")
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
