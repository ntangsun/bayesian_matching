#!/usr/bin/env python3
"""
experiments/beta_regularization_path_replications.py

python experiments/beta_regularization_path_replications.py `
    --n 300 `
    --out-detail results/beta_regularization_path_replications_N300_detail.csv `
    --out-summary results/beta_regularization_path_replications_N300_summary.csv `
    --out-plot results/beta_regularization_path_replications_N300.png

python experiments/beta_regularization_path_replications.py `
    --n 300 `
    --n-reps 5 `
    --individual-reps 5 `
    --print-every 1 `
    --out-detail results/beta_regularization_path_replications_N300_first5_detail.csv `
    --out-summary results/beta_regularization_path_replications_N300_first5_summary.csv `
    --out-plot results/beta_regularization_path_replications_N300_first5.png `
    --out-individual-plot results/beta_regularization_path_replications_N300_individual_first5.png

Replicated diagnostic for finite-beta Bayesian unconstrained 1-to-1 matching.

This extends beta_regularization_path_one_dataset.py from a single fixed
dataset to many generated replications. The experiment studies how the
posterior-averaged finite-beta estimator moves toward the deterministic
nearest-neighbor estimator as beta increases.
"""

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_matching_from_datasets import (
    distance_matrix,
    lm_slope_variance,
    recipient_donor_indices,
)


TRUE_TAU = 1.0
DEFAULT_BETAS = "0,0.25,0.5,1,2,3,4,5,8,10,15,20,30,50,75,100,150,200"


def parse_betas(value):
    """Parse comma-separated finite beta values."""
    betas = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        beta = float(token)
        if beta < 0:
            raise ValueError("beta values must be nonnegative.")
        betas.append(beta)
    if not betas:
        raise ValueError("betas must contain at least one value.")
    return betas


def beta_label(beta):
    """Stable beta label for CSV and plot ticks."""
    if float(beta).is_integer():
        return str(int(beta))
    return str(beta)


def create_one_dataset(n, recipient_frac, mu, rng):
    """
    Generate one nonlinear causal-inference dataset.

    The nonlinear response uses the table-scale version
    g0(X) = 18 + exp(X @ b) / 10, which keeps outcomes on the same scale as
    the reported simulation tables and the linear response surface.
    """
    n_rc = int(recipient_frac * n)
    n_dn = n - n_rc
    if n_rc <= 0 or n_dn <= 0:
        raise ValueError("recipient_frac and n must create both recipients and donors.")

    g = np.concatenate([np.ones(n_rc, dtype=int), np.zeros(n_dn, dtype=int)])

    p = 4
    X_rc = rng.normal(mu, 1, size=(n_rc, p))
    X_dn = rng.normal(0, 1, size=(n_dn, p))
    X = np.vstack([X_rc, X_dn])

    b = np.array([0.1, 0.2, 0.2, 0.1])
    g0 = 18 + np.exp(X @ b) / 10
    mean_y = g0 + g
    Y = rng.normal(mean_y, 1)

    return X, Y, g, g0


def tau_from_match_cols(Y, g, match_cols):
    """Compute tau using the project's complete-data regression convention."""
    rc, dn = recipient_donor_indices(g)
    Y_rc = Y[rc]
    Y_match = Y[dn[match_cols]]
    Y_oc = np.concatenate([Y_rc, Y_match])
    g_oc = np.concatenate([np.ones(len(rc)), np.zeros(len(rc))])
    tau_hat, _ = lm_slope_variance(Y_oc, g_oc)
    return float(tau_hat)


def gap_summaries(X, g, g0, match_cols):
    """Compute mean g0-gap and mean covariate gap for one matching draw."""
    rc, dn = recipient_donor_indices(g)
    matched_dn = dn[match_cols]
    return {
        "g0_gap": float((g0[rc] - g0[matched_dn]).mean()),
        "x_gap": float((X[rc] - X[matched_dn]).mean()),
    }


def nearest_neighbor_baseline(X, Y, g, g0):
    """Compute deterministic nearest-neighbor unconstrained 1-to-1 estimates."""
    dist = distance_matrix(X, g, 2)
    match_cols = np.argmin(dist, axis=1)
    tau_hat = tau_from_match_cols(Y, g, match_cols)
    gaps = gap_summaries(X, g, g0, match_cols)
    return {
        "tau_hat_NN": tau_hat,
        "abs_bias_NN": abs(tau_hat - TRUE_TAU),
        "g0_gap_NN": gaps["g0_gap"],
        "x_gap_NN": gaps["x_gap"],
    }


def posterior_probabilities(dist, beta):
    """
    Build row-wise donor probabilities using the project convention.

    alpha is proportional to 1 / dist_df**beta, where dist_df is shifted by the
    row minimum as in run_matching_from_datasets.py.
    """
    offset = dist.min(axis=1)
    dist_df = dist + offset[:, None]

    if beta == 0:
        return np.full_like(dist_df, 1 / dist_df.shape[1], dtype=float)

    log_weight = -beta * np.log(dist_df)
    log_weight -= log_weight.max(axis=1, keepdims=True)
    alpha = np.exp(log_weight)
    alpha /= alpha.sum(axis=1, keepdims=True)
    return alpha


def run_finite_beta(X, Y, g, g0, beta, n_mcmc, rng):
    """Average tau and balance diagnostics over finite-beta posterior draws."""
    rc, dn = recipient_donor_indices(g)
    n_rc = len(rc)

    dist = distance_matrix(X, g, 2)
    alpha = posterior_probabilities(dist, beta)
    alpha_cdf = np.cumsum(alpha, axis=1)
    alpha_cdf[:, -1] = 1.0

    Y_rc = Y[rc]
    g_oc = np.concatenate([np.ones(n_rc), np.zeros(n_rc)])

    tau_sum = 0.0
    g0_gap_sum = 0.0
    x_gap_sum = 0.0

    for _ in range(n_mcmc):
        u = rng.random(n_rc)
        match_cols = (u[:, None] > alpha_cdf).sum(axis=1)
        matched_dn = dn[match_cols]

        Y_match = Y[matched_dn]
        Y_oc = np.concatenate([Y_rc, Y_match])
        tau_hat, _ = lm_slope_variance(Y_oc, g_oc)

        tau_sum += tau_hat
        g0_gap_sum += (g0[rc] - g0[matched_dn]).mean()
        x_gap_sum += (X[rc] - X[matched_dn]).mean()

    tau_hat_beta = float(tau_sum / n_mcmc)
    return {
        "tau_hat": tau_hat_beta,
        "abs_bias": abs(tau_hat_beta - TRUE_TAU),
        "g0_gap": float(g0_gap_sum / n_mcmc),
        "x_gap": float(x_gap_sum / n_mcmc),
    }


def summarize_rows(detail_rows, betas):
    """Summarize detail rows across replications for each beta."""
    summary_rows = []
    for beta in betas:
        label = beta_label(beta)
        group = [row for row in detail_rows if row["beta"] == label]
        n_reps = len(group)
        if n_reps == 0:
            continue

        def mean(col):
            return float(np.mean([row[col] for row in group]))

        def se(col):
            vals = np.array([row[col] for row in group], dtype=float)
            if len(vals) <= 1:
                return 0.0
            return float(vals.std(ddof=1) / math.sqrt(len(vals)))

        summary_rows.append(
            {
                "beta": label,
                "n_reps": n_reps,
                "mean_tau_hat": mean("tau_hat"),
                "se_tau_hat": se("tau_hat"),
                "mean_abs_bias": mean("abs_bias"),
                "se_abs_bias": se("abs_bias"),
                "mean_g0_gap": mean("g0_gap"),
                "se_g0_gap": se("g0_gap"),
                "mean_x_gap": mean("x_gap"),
                "se_x_gap": se("x_gap"),
                "mean_tau_hat_NN": mean("tau_hat_NN"),
                "mean_abs_bias_NN": mean("abs_bias_NN"),
                "mean_g0_gap_NN": mean("g0_gap_NN"),
                "mean_x_gap_NN": mean("x_gap_NN"),
                "mean_distance_to_NN": mean("distance_to_NN"),
                "se_distance_to_NN": se("distance_to_NN"),
                "n": group[0]["n"],
                "n_mcmc": group[0]["n_mcmc"],
                "mu": group[0]["mu"],
                "seed": group[0]["seed"],
            }
        )
    return summary_rows


def write_csv(rows, out_csv, fieldnames):
    """Save rows to CSV."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows, out_plot):
    """Create replicated beta regularization path plot."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(summary_rows))
    beta_ticks = [row["beta"] for row in summary_rows]

    tau = np.array([row["mean_tau_hat"] for row in summary_rows])
    tau_se = np.array([row["se_tau_hat"] for row in summary_rows])
    abs_bias = np.array([row["mean_abs_bias"] for row in summary_rows])
    abs_bias_se = np.array([row["se_abs_bias"] for row in summary_rows])
    dist_nn = np.array([row["mean_distance_to_NN"] for row in summary_rows])
    dist_nn_se = np.array([row["se_distance_to_NN"] for row in summary_rows])

    mean_tau_nn = summary_rows[0]["mean_tau_hat_NN"]
    mean_abs_bias_nn = summary_rows[0]["mean_abs_bias_NN"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(x, tau, marker="o", linewidth=2, label="Finite beta")
    axes[0].fill_between(x, tau - 1.96 * tau_se, tau + 1.96 * tau_se, alpha=0.16)
    axes[0].axhline(TRUE_TAU, color="black", linestyle="--", linewidth=1, label="True tau")
    axes[0].axhline(mean_tau_nn, color="#d62728", linestyle=":", linewidth=2, label="Mean NN")
    axes[0].set_ylabel("Mean tau_hat")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, abs_bias, marker="o", linewidth=2, color="#2ca02c")
    axes[1].fill_between(
        x,
        abs_bias - 1.96 * abs_bias_se,
        abs_bias + 1.96 * abs_bias_se,
        color="#2ca02c",
        alpha=0.16,
    )
    axes[1].axhline(mean_abs_bias_nn, color="#d62728", linestyle=":", linewidth=2, label="Mean NN abs bias")
    axes[1].set_ylabel("Mean |tau_hat - 1|")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, dist_nn, marker="o", linewidth=2, color="#9467bd")
    axes[2].fill_between(
        x,
        np.maximum(dist_nn - 1.96 * dist_nn_se, 0),
        dist_nn + 1.96 * dist_nn_se,
        color="#9467bd",
        alpha=0.16,
    )
    axes[2].set_ylabel("Mean |tau_hat - tau_hat_NN|")
    axes[2].set_xlabel("beta")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(beta_ticks, rotation=35)
    axes[2].grid(alpha=0.25)

    fig.suptitle("Beta Regularization Path Across Generated Datasets")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def plot_individual_rep_paths(detail_rows, betas, n_reps_to_plot, out_plot):
    """Plot tau_hat beta paths for the first few individual replications."""
    if n_reps_to_plot <= 0:
        return

    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    reps = sorted({row["rep"] for row in detail_rows})[:n_reps_to_plot]
    if not reps:
        return

    beta_ticks = [beta_label(beta) for beta in betas]
    x = np.arange(len(beta_ticks))

    fig, axes = plt.subplots(
        len(reps),
        1,
        figsize=(11, max(3, 2.4 * len(reps))),
        sharex=True,
    )
    if len(reps) == 1:
        axes = [axes]

    for ax, rep in zip(axes, reps):
        rep_rows = [row for row in detail_rows if row["rep"] == rep]
        rep_by_beta = {row["beta"]: row for row in rep_rows}
        y = np.array([rep_by_beta[label]["tau_hat"] for label in beta_ticks])
        tau_nn = rep_rows[0]["tau_hat_NN"]

        ax.plot(x, y, marker="o", linewidth=2, label="Finite beta")
        ax.axhline(TRUE_TAU, color="black", linestyle="--", linewidth=1, label="True tau")
        ax.axhline(tau_nn, color="#d62728", linestyle=":", linewidth=2, label="NN")
        ax.set_ylabel(f"rep {rep}\ntau_hat")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("beta")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(beta_ticks, rotation=35)

    fig.suptitle("Single-Replication Beta Paths")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--recipient-frac", type=float, default=0.3)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--n-mcmc", type=int, default=1000)
    parser.add_argument("--betas", type=str, default=DEFAULT_BETAS)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--individual-reps",
        type=int,
        default=5,
        help="Number of first replications to include in the individual tau path plot.",
    )
    parser.add_argument(
        "--out-detail",
        type=str,
        default="results/beta_regularization_path_replications_detail.csv",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="results/beta_regularization_path_replications_summary.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="results/beta_regularization_path_replications.png",
    )
    parser.add_argument(
        "--out-individual-plot",
        type=str,
        default="results/beta_regularization_path_replications_individual_first5.png",
    )
    args = parser.parse_args()

    if args.n_reps <= 0:
        raise ValueError("n-reps must be positive.")
    if args.n_mcmc <= 0:
        raise ValueError("n-mcmc must be positive.")

    betas = parse_betas(args.betas)
    detail_rows = []

    for rep in range(1, args.n_reps + 1):
        if rep == 1 or rep % args.print_every == 0 or rep == args.n_reps:
            print(f"Replication {rep}/{args.n_reps}")

        data_rng = np.random.default_rng(args.seed + rep)
        X, Y, g, g0 = create_one_dataset(
            n=args.n,
            recipient_frac=args.recipient_frac,
            mu=args.mu,
            rng=data_rng,
        )
        nn = nearest_neighbor_baseline(X, Y, g, g0)

        for beta_pos, beta in enumerate(betas):
            draw_seed = args.seed + 1_000_000 + rep * 10_000 + beta_pos
            draw_rng = np.random.default_rng(draw_seed)
            result = run_finite_beta(
                X=X,
                Y=Y,
                g=g,
                g0=g0,
                beta=beta,
                n_mcmc=args.n_mcmc,
                rng=draw_rng,
            )

            detail_rows.append(
                {
                    "rep": rep,
                    "beta": beta_label(beta),
                    **result,
                    **nn,
                    "distance_to_NN": abs(result["tau_hat"] - nn["tau_hat_NN"]),
                    "n": args.n,
                    "n_mcmc": args.n_mcmc,
                    "mu": args.mu,
                    "seed": args.seed,
                }
            )

    summary_rows = summarize_rows(detail_rows, betas)

    detail_fields = [
        "rep",
        "beta",
        "tau_hat",
        "abs_bias",
        "g0_gap",
        "x_gap",
        "tau_hat_NN",
        "abs_bias_NN",
        "g0_gap_NN",
        "x_gap_NN",
        "distance_to_NN",
        "n",
        "n_mcmc",
        "mu",
        "seed",
    ]
    summary_fields = [
        "beta",
        "n_reps",
        "mean_tau_hat",
        "se_tau_hat",
        "mean_abs_bias",
        "se_abs_bias",
        "mean_g0_gap",
        "se_g0_gap",
        "mean_x_gap",
        "se_x_gap",
        "mean_tau_hat_NN",
        "mean_abs_bias_NN",
        "mean_g0_gap_NN",
        "mean_x_gap_NN",
        "mean_distance_to_NN",
        "se_distance_to_NN",
        "n",
        "n_mcmc",
        "mu",
        "seed",
    ]

    write_csv(detail_rows, args.out_detail, detail_fields)
    write_csv(summary_rows, args.out_summary, summary_fields)
    plot_summary(summary_rows, args.out_plot)
    plot_individual_rep_paths(
        detail_rows,
        betas,
        args.individual_reps,
        args.out_individual_plot,
    )

    best = min(summary_rows, key=lambda row: row["mean_abs_bias"])
    print(
        "Best beta by mean absolute bias: "
        f"beta={best['beta']}, mean_abs_bias={best['mean_abs_bias']:.6f}"
    )
    print(f"Saved detail CSV to {args.out_detail}")
    print(f"Saved summary CSV to {args.out_summary}")
    print(f"Saved plot to {args.out_plot}")
    if args.individual_reps > 0:
        print(f"Saved individual-rep plot to {args.out_individual_plot}")


if __name__ == "__main__":
    main()
