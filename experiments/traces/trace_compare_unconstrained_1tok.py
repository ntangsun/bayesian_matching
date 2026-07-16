#!/usr/bin/env python3
r"""

python experiments/traces/trace_compare_unconstrained_1tok.py `
    --datasets datasets_sb01.npz `
    --n-mcmc 5000 `
    --checkpoint-every 250 `
    --max-sim 100 `
    --out-detail results\experiments\mcmc_vs_mc\unconstrained_1tok_mcmc_vs_mc_trace_by_sim.csv `
    --out-summary results\experiments\mcmc_vs_mc\unconstrained_1tok_mcmc_vs_mc_trace_summary.csv `
    --out-plot results\experiments\mcmc_vs_mc\unconstrained_1tok_mcmc_vs_mc_trace.png

experiments/traces/trace_compare_unconstrained_1tok.py

Compare the original Metropolis row-swap sampler with the direct Monte Carlo
dynamic-programming sampler for unconstrained 1-to-k Bayesian matching.

The diagnostic is intentionally estimator-focused: for each simulation
replication, both samplers update the running mean of the same tau estimator at
matching checkpoints. If both samplers target the same posterior distribution,
their running tau summaries should approach each other as the number of
imputations grows.

Example:
    python experiments/traces/trace_compare_unconstrained_1tok.py --datasets datasets_sb01.npz --max-sim 100
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
    lm_slope_variance,
    recipient_donor_indices,
)


def setup_common(X, Y, g, beta, k):
    """Build objects shared by both unconstrained 1-to-k samplers."""
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)
    n_dn = len(dn)

    if not 1 <= k < n_dn:
        raise ValueError("k must be at least 1 and smaller than the number of donors.")

    dist = distance_matrix(X, g, 2)
    offset = dist.min(axis=1)
    dist_df = dist + offset[:, None]

    # Keep the project convention used in the main runner: p = beta.
    alpha = 1 / (dist_df ** beta)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    Y_rc = Y[rc]
    Y_matches = np.empty((n_rc, k))
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(k * n_rc)])

    return rc, dn, alpha, Y_rc, Y_matches, g_oc


def tau_from_matches(Y_rc, Y_matches, g_oc):
    """Compute the same complete-data tau estimator used by the main methods."""
    Y_match = Y_matches.ravel()
    Y_oc = np.concatenate([Y_rc, Y_match])
    tau_hat, _ = lm_slope_variance(Y_oc, g_oc)
    return tau_hat


def build_mc_dp(alpha, k):
    """
    Build the dynamic-programming table for direct row sampling.

    This is done once per simulation replication. Each later Monte Carlo draw
    scans the donor positions using this fixed table.
    """
    n_rc, n_dn = alpha.shape
    row_weights = alpha / (1 - alpha)

    dp = np.zeros((n_rc, n_dn + 1, k + 1))
    dp[:, :, 0] = 1.0

    for donor_pos in range(n_dn - 1, -1, -1):
        for kk in range(1, k + 1):
            dp[:, donor_pos, kk] = (
                dp[:, donor_pos + 1, kk]
                + row_weights[:, donor_pos] * dp[:, donor_pos + 1, kk - 1]
            )

    return row_weights, dp


def sample_mcmc_iteration(Y, dn, alpha, donor_order, Y_matches, rng, k):
    """Advance the Metropolis row-swap sampler by one full iteration."""
    n_rc, n_dn = alpha.shape
    accepted_any = False
    accepted_row_updates = 0

    for r in range(n_rc):
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

    return accepted_any, accepted_row_updates


def sample_mc_iteration(Y, dn, row_weights, dp, Y_matches, rng, k):
    """Draw one independent matching structure with the DP row sampler."""
    n_rc, n_dn = row_weights.shape
    match_cols = np.empty((n_rc, k), dtype=int)

    for r in range(n_rc):
        remaining = k
        selected = 0

        for donor_pos in range(n_dn):
            if remaining == 0:
                break

            denom = dp[r, donor_pos, remaining]
            numer = (
                row_weights[r, donor_pos]
                * dp[r, donor_pos + 1, remaining - 1]
            )
            include_prob = numer / denom

            if rng.random() < include_prob:
                match_cols[r, selected] = donor_pos
                selected += 1
                remaining -= 1

        if remaining != 0:
            raise RuntimeError("Monte Carlo row sampler failed to select k donors.")

        Y_matches[r, :] = Y[dn[match_cols[r, :]]]


def trace_one_sampler(
    sampler,
    X,
    Y,
    g,
    seed,
    sim_number,
    n_mcmc,
    checkpoint_every,
    beta,
    k,
):
    """Run one sampler on one replication and return checkpoint rows."""
    rc, dn, alpha, Y_rc, Y_matches, g_oc = setup_common(X, Y, g, beta, k)
    rng = np.random.default_rng(int(seed))

    tau_sum = 0.0
    accepted_iterations = 0
    accepted_row_updates = 0
    rows = []

    if sampler == "mcmc":
        donor_order = np.argsort(distance_matrix(X, g, 2), axis=1)
    elif sampler == "monte_carlo":
        row_weights, dp = build_mc_dp(alpha, k)
    else:
        raise ValueError("Unknown sampler.")

    for mcmc_iter in range(1, n_mcmc + 1):
        if sampler == "mcmc":
            accepted_any, row_updates = sample_mcmc_iteration(
                Y, dn, alpha, donor_order, Y_matches, rng, k
            )
            if accepted_any:
                accepted_iterations += 1
            accepted_row_updates += row_updates
        else:
            sample_mc_iteration(Y, dn, row_weights, dp, Y_matches, rng, k)

        tau_hat = tau_from_matches(Y_rc, Y_matches, g_oc)
        tau_sum += tau_hat
        running_tau = tau_sum / mcmc_iter

        if mcmc_iter % checkpoint_every == 0 or mcmc_iter == n_mcmc:
            row = {
                "sim": sim_number,
                "sampler": sampler,
                "mcmc_iter": mcmc_iter,
                "tau_current": tau_hat,
                "running_tau": running_tau,
            }
            if sampler == "mcmc":
                row["accepted_iteration_rate"] = accepted_iterations / mcmc_iter
                row["accepted_row_update_rate"] = (
                    accepted_row_updates / (mcmc_iter * len(rc))
                )
            rows.append(row)

    return rows


def summarize_detail(detail_df, true_tau):
    """Average running tau checkpoints across replications and samplers."""
    summary = (
        detail_df.groupby(["sampler", "mcmc_iter"], as_index=False)
        .agg(
            n_sim=("sim", "nunique"),
            mean_tau=("running_tau", "mean"),
            sd_tau=("running_tau", "std"),
            q05_tau=("running_tau", lambda x: x.quantile(0.05)),
            q95_tau=("running_tau", lambda x: x.quantile(0.95)),
        )
    )
    summary["bias"] = summary["mean_tau"] - true_tau
    summary["abs_bias"] = summary["bias"].abs()
    summary["se_tau"] = summary["sd_tau"] / np.sqrt(summary["n_sim"])

    wide = summary.pivot(index="mcmc_iter", columns="sampler", values="mean_tau")
    if {"mcmc", "monte_carlo"}.issubset(wide.columns):
        diff = (
            wide["mcmc"]
            .subtract(wide["monte_carlo"])
            .rename("mean_tau_diff_mcmc_minus_mc")
            .reset_index()
        )
        summary = summary.merge(diff, on="mcmc_iter", how="left")

    return summary


def make_comparison(detail_df):
    """Create one row per checkpoint with paired running tau differences."""
    wide = (
        detail_df.pivot_table(
            index=["sim", "mcmc_iter"],
            columns="sampler",
            values="running_tau",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    if {"mcmc", "monte_carlo"}.issubset(wide.columns):
        wide["tau_diff_mcmc_minus_mc"] = wide["mcmc"] - wide["monte_carlo"]

    return wide.rename(
        columns={
            "mcmc": "running_tau_mcmc",
            "monte_carlo": "running_tau_monte_carlo",
        }
    )


def plot_summary(summary_df, true_tau, out_plot):
    """Plot running tau for both samplers and their mean difference."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {"mcmc": "#1f77b4", "monte_carlo": "#2ca02c"}
    labels = {"mcmc": "MCMC row-swap", "monte_carlo": "Monte Carlo DP"}

    for sampler, group in summary_df.groupby("sampler", sort=False):
        group = group.sort_values("mcmc_iter")
        x = group["mcmc_iter"].to_numpy()
        mean_tau = group["mean_tau"].to_numpy()
        q05 = group["q05_tau"].to_numpy()
        q95 = group["q95_tau"].to_numpy()

        axes[0].plot(
            x,
            mean_tau,
            color=colors.get(sampler),
            linewidth=2,
            label=labels.get(sampler, sampler),
        )
        axes[0].fill_between(x, q05, q95, color=colors.get(sampler), alpha=0.12)

    axes[0].axhline(true_tau, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Running mean tau")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    diff_df = (
        summary_df[["mcmc_iter", "mean_tau_diff_mcmc_minus_mc"]]
        .dropna()
        .drop_duplicates()
        .sort_values("mcmc_iter")
    )
    axes[1].plot(
        diff_df["mcmc_iter"],
        diff_df["mean_tau_diff_mcmc_minus_mc"],
        color="#d62728",
        linewidth=2,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of imputations")
    axes[1].set_ylabel("Mean tau difference")
    axes[1].grid(alpha=0.25)

    fig.suptitle("MCMC vs Direct Monte Carlo: Unconstrained 1-to-k Tau Trace")
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
    parser.add_argument("--max-sim", type=int, default=100)
    parser.add_argument(
        "--sim-index",
        type=int,
        default=None,
        help="One-based dataset replication to fix. If provided, only this replication is traced.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--out-detail",
        type=str,
        default="results/experiments/mcmc_vs_mc/unconstrained_1tok_mcmc_vs_mc_trace_by_sim.csv",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="results/experiments/mcmc_vs_mc/unconstrained_1tok_mcmc_vs_mc_trace_summary.csv",
    )
    parser.add_argument(
        "--out-comparison",
        type=str,
        default="results/experiments/mcmc_vs_mc/unconstrained_1tok_mcmc_vs_mc_trace_comparison.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="results/experiments/mcmc_vs_mc/unconstrained_1tok_mcmc_vs_mc_trace.png",
    )
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive.")

    data = np.load(args.datasets)
    X_all = data["X"]
    Y_all = data["Y"]
    g_all = data["g"]
    seeds = data["seeds"] if "seeds" in data else np.arange(1, len(X_all) + 1)

    if args.sim_index is not None:
        if not 1 <= args.sim_index <= len(X_all):
            raise ValueError("sim-index must be between 1 and the number of datasets.")
        sim_indices = [args.sim_index - 1]
    else:
        n_sim = min(len(X_all), args.max_sim)
        sim_indices = list(range(n_sim))

    all_rows = []

    for pos, s in enumerate(sim_indices, start=1):
        if pos == 1 or pos % args.print_every == 0 or pos == len(sim_indices):
            print(f"Compare trace simulation {pos}/{len(sim_indices)} (dataset {s + 1})")

        # Use independent but deterministic streams for the two algorithms.
        mcmc_seed = int(seeds[s]) + 1_000_000
        mc_seed = int(seeds[s]) + 2_000_000

        for sampler, sampler_seed in [
            ("mcmc", mcmc_seed),
            ("monte_carlo", mc_seed),
        ]:
            all_rows.extend(
                trace_one_sampler(
                    sampler,
                    X_all[s],
                    Y_all[s],
                    g_all[s],
                    sampler_seed,
                    sim_number=s + 1,
                    n_mcmc=args.n_mcmc,
                    checkpoint_every=args.checkpoint_every,
                    beta=args.beta,
                    k=args.k,
                )
            )

    detail_df = pd.DataFrame(all_rows)
    summary_df = summarize_detail(detail_df, true_tau=args.true_tau)
    comparison_df = make_comparison(detail_df)

    out_detail = Path(args.out_detail)
    out_summary = Path(args.out_summary)
    out_comparison = Path(args.out_comparison)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_comparison.parent.mkdir(parents=True, exist_ok=True)

    detail_df.to_csv(out_detail, index=False)
    summary_df.to_csv(out_summary, index=False)
    comparison_df.to_csv(out_comparison, index=False)
    plot_summary(summary_df, true_tau=args.true_tau, out_plot=args.out_plot)

    print(f"Saved detail trace to {out_detail}")
    print(f"Saved summary trace to {out_summary}")
    print(f"Saved comparison trace to {out_comparison}")
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
