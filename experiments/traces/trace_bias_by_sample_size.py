#!/usr/bin/env python3
"""
experiments/traces/trace_bias_by_sample_size.py

Study whether the finite-sample bias of Bayesian unconstrained 1-to-k matching
decreases as the dataset size N increases.

The script generates fresh datasets for each sample size, estimates tau within
each replication, then summarizes Monte Carlo bias separately for each N.

Pilot example:
    python experiments/traces/trace_bias_by_sample_size.py --betas 1
"""

import argparse
import csv
import math
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


def parse_sample_sizes(value):
    """Parse comma-separated sample sizes such as '100,300,1000,3000'."""
    sizes = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not sizes:
        raise ValueError("sample-sizes must contain at least one positive integer.")
    if any(x <= 0 for x in sizes):
        raise ValueError("sample-sizes must be positive.")
    return sizes


def parse_betas(value):
    """Parse comma-separated betas, with 'inf' for deterministic nearest-k."""
    betas = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"inf", "infinity"}:
            betas.append(math.inf)
        else:
            beta = float(token)
            if beta <= 0:
                raise ValueError("finite beta values must be positive.")
            betas.append(beta)
    if not betas:
        raise ValueError("betas must contain at least one value.")
    return betas


def parse_reps_by_size(value):
    """Parse mappings such as '100:300,200:300,1000:120'."""
    if value is None or not value.strip():
        return {}

    mapping = {}
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError("reps-by-size entries must look like N:reps.")
        n_raw, reps_raw = token.split(":", 1)
        n_pop = int(n_raw.strip())
        reps = int(reps_raw.strip())
        if n_pop <= 0 or reps <= 0:
            raise ValueError("reps-by-size keys and values must be positive.")
        mapping[n_pop] = reps
    return mapping


def beta_label(beta):
    """Stable string label for CSV/plot output."""
    if math.isinf(beta):
        return "inf"
    if float(beta).is_integer():
        return str(int(beta))
    return str(beta)


def create_causal_data(n_pop, recipient_frac, mu, rng):
    """
    Generate one causal-inference dataset using the Bayesian Matching DGP.

    g = 1 for recipients/treated and 0 for donors/controls.
    The true ATT is 1 because g1(X) = g0(X) + 1.
    """
    n_rc = int(recipient_frac * n_pop)
    n_dn = n_pop - n_rc
    if n_rc <= 0 or n_dn <= 0:
        raise ValueError("recipient_frac and N must create both recipients and donors.")

    g = np.concatenate([np.ones(n_rc, dtype=int), np.zeros(n_dn, dtype=int)])

    x_rc = rng.normal(mu, 1, size=(n_rc, 4))
    x_dn = rng.normal(0, 1, size=(n_dn, 4))
    X = np.vstack([x_rc, x_dn])

    coeff = np.array([0.1, 0.2, 0.2, 0.1])
    g0 = 18 + X @ coeff
    mean_y = g0 + g
    Y = rng.normal(mean_y, 1)

    return X, Y, g


def estimate_tau_from_matches(Y, g, matched_y):
    """
    Estimate tau using the same outcome regression convention as the main code.

    matched_y is an n_rc by k matrix of imputed donor outcomes.
    """
    rc, _ = recipient_donor_indices(g)
    Y_rc = Y[rc]
    Y_match = matched_y.ravel()
    Y_oc = np.concatenate([Y_rc, Y_match])
    g_oc = np.concatenate([np.ones(len(Y_rc)), np.zeros(len(Y_match))])
    tau_hat, _ = lm_slope_variance(Y_oc, g_oc)
    return tau_hat


def mean_covariate_gap(X, g, matched_x):
    """
    Average recipient-minus-matched-donor covariate gap.

    matched_x is n_rc by k by 4. We average matched donors for each recipient,
    subtract from recipient covariates, then average over recipients and the
    four covariates.
    """
    rc, _ = recipient_donor_indices(g)
    recipient_x = X[rc]
    donor_mean_x = matched_x.mean(axis=1)
    return float((recipient_x - donor_mean_x).mean())


def estimate_tau_nearest_k(X, Y, g, k):
    """Deterministic nearest-neighbor 1-to-k baseline for beta = inf."""
    rc, dn = recipient_donor_indices(g)
    dist = distance_matrix(X, g, 2)
    nearest = np.argsort(dist, axis=1)[:, :k]

    matched_y = Y[dn[nearest]]
    matched_x = X[dn[nearest]]

    return {
        "tau_hat": estimate_tau_from_matches(Y, g, matched_y),
        "mean_x_gap": mean_covariate_gap(X, g, matched_x),
    }


def setup_finite_beta_sampler(X, Y, g, beta, k):
    """
    Build fixed objects for the finite-beta unconstrained 1-to-k sampler.

    donor_order[r, :k] are selected donors for recipient r.
    donor_order[r, k:] are unselected donors for recipient r.
    """
    rc, dn = recipient_donor_indices(g)
    n_dn = len(dn)
    if not 1 <= k < n_dn:
        raise ValueError("k must be at least 1 and smaller than the number of donors.")

    dist = distance_matrix(X, g, 2)
    offset = dist.min(axis=1)
    dist_df = dist + offset[:, None]

    alpha = 1 / (dist_df ** beta)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    donor_order = np.argsort(dist, axis=1)
    matched_y = np.empty((len(rc), k))
    matched_x = np.empty((len(rc), k, X.shape[1]))

    return rc, dn, alpha, donor_order, matched_y, matched_x


def estimate_tau_finite_beta(X, Y, g, beta, k, n_mcmc, rng):
    """
    Estimate tau by averaging over finite-beta posterior matching draws.

    tau and covariate gap are accumulated online so we do not store all MCMC
    draws.
    """
    rc, dn, alpha, donor_order, matched_y, matched_x = setup_finite_beta_sampler(
        X, Y, g, beta, k
    )

    n_rc = len(rc)
    n_dn = len(dn)
    tau_sum = 0.0
    x_gap_sum = 0.0

    for _ in range(n_mcmc):
        # -----------------------------
        # MCMC Sampling
        # -----------------------------
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

            selected = donor_order[r, :k]
            matched_y[r, :] = Y[dn[selected]]
            matched_x[r, :, :] = X[dn[selected]]

        # -----------------------------
        # Online Estimate Update
        # -----------------------------
        tau_sum += estimate_tau_from_matches(Y, g, matched_y)
        x_gap_sum += mean_covariate_gap(X, g, matched_x)

    return {
        "tau_hat": tau_sum / n_mcmc,
        "mean_x_gap": x_gap_sum / n_mcmc,
    }


def summarize_replications(tau_hats, x_gaps, true_tau):
    """Summarize tau and covariate-gap values across replications."""
    tau_hats = np.asarray(tau_hats, dtype=float)
    x_gaps = np.asarray(x_gaps, dtype=float)
    mean_tau = float(tau_hats.mean())
    bias = mean_tau - true_tau
    sd_tau = float(tau_hats.std(ddof=1)) if len(tau_hats) > 1 else 0.0
    return {
        "mean_tau": mean_tau,
        "bias": bias,
        "abs_bias": abs(bias),
        "sd_tau": sd_tau,
        "mc_se_bias": sd_tau / math.sqrt(len(tau_hats)),
        "mean_x_gap": float(x_gaps.mean()),
    }


def plot_results(results_df, out_plot, log_x=False):
    """Plot bias and covariate gap against sample size, one line per beta."""
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for beta, group in results_df.groupby("beta", sort=False):
        group = group.sort_values("N")
        axes[0].plot(
            group["N"],
            group["bias"],
            marker="o",
            linewidth=2,
            label=f"beta={beta}",
        )
        axes[1].plot(
            group["N"],
            group["mean_x_gap"],
            marker="o",
            linewidth=2,
            label=f"beta={beta}",
        )

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Bias")
    axes[0].set_title("Bias vs Sample Size for Unconstrained 1-to-k Matching")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Sample size N")
    axes[1].set_ylabel("Mean covariate gap")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    if log_x:
        axes[1].set_xscale("log")
    else:
        ticks = sorted(results_df["N"].unique())
        axes[1].set_xticks(ticks)
        axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(out_plot, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=str, default="100,300,1000,3000")
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument(
        "--reps-by-size",
        type=str,
        default="",
        help="Optional per-N reps mapping, e.g. 100:300,200:300,1000:120",
    )
    parser.add_argument("--n-mcmc", type=int, default=300)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--betas", type=str, default="1")
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--recipient-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--true-tau", type=float, default=1.0)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/experiments/bias_by_sample_size/bias_by_sample_size.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="results/experiments/bias_by_sample_size/bias_by_sample_size.png",
    )
    parser.add_argument("--log-x", action="store_true", help="Use log scale for sample-size axis")
    args = parser.parse_args()

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    betas = parse_betas(args.betas)
    reps_by_size = parse_reps_by_size(args.reps_by_size)
    rng = np.random.default_rng(args.seed)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "beta",
        "N",
        "n_reps",
        "n_mcmc",
        "k",
        "mu",
        "recipient_frac",
        "mean_tau",
        "bias",
        "abs_bias",
        "sd_tau",
        "mc_se_bias",
        "mean_x_gap",
    ]

    rows = []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for beta in betas:
            beta_name = beta_label(beta)
            for n_pop in sample_sizes:
                n_reps = reps_by_size.get(n_pop, args.n_reps)
                print(
                    f"Running beta={beta_name}, N={n_pop}, "
                    f"reps={n_reps}, n_mcmc={args.n_mcmc}"
                )

                tau_hats = []
                x_gaps = []
                for rep in range(1, n_reps + 1):
                    if rep == 1 or rep % 25 == 0 or rep == n_reps:
                        print(f"  beta={beta_name} N={n_pop} replication {rep}/{n_reps}")

                    X, Y, g = create_causal_data(
                        n_pop=n_pop,
                        recipient_frac=args.recipient_frac,
                        mu=args.mu,
                        rng=rng,
                    )

                    if math.isinf(beta):
                        result = estimate_tau_nearest_k(X, Y, g, args.k)
                    else:
                        result = estimate_tau_finite_beta(
                            X, Y, g, beta, args.k, args.n_mcmc, rng
                        )

                    tau_hats.append(result["tau_hat"])
                    x_gaps.append(result["mean_x_gap"])

                summary = summarize_replications(tau_hats, x_gaps, args.true_tau)
                row = {
                    "beta": beta_name,
                    "N": n_pop,
                    "n_reps": n_reps,
                    "n_mcmc": args.n_mcmc,
                    "k": args.k,
                    "mu": args.mu,
                    "recipient_frac": args.recipient_frac,
                    **summary,
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()

                print(
                    f"Finished beta={beta_name}, N={n_pop}: "
                    f"bias={row['bias']:.6f}, mean_x_gap={row['mean_x_gap']:.6f}"
                )

    results_df = pd.DataFrame(rows)
    plot_results(results_df, args.out_plot, log_x=args.log_x)

    print(f"Saved CSV to {out_csv}")
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
