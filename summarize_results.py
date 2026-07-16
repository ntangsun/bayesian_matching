#!/usr/bin/env python3
"""
summarize_results.py

Summarize result CSV files and print a compact readable table.

Features:
1. Summarize one specific file:
      python summarize_results.py --results-file results/matching/unconstrained_1to1/unconstrained_1to1_beta1_2.csv

2. Summarize files matching a pattern:
      python summarize_results.py --results-dir results/matching/unconstrained_1to1 --pattern "unconstrained_1to1_beta1*.csv"

3. By default, DOES NOT save a summary CSV.
   It only prints the compact summary.

4. To save, add --save.
   If --out is not provided, it auto-saves with incrementing filenames:
      summaries/summary_1.csv
      summaries/summary_2.csv
      ...

5. To save with a custom prefix:
      python summarize_results.py --results-dir results/matching/unconstrained_1to1 --pattern "unconstrained_1to1_beta1*.csv" --save --out-prefix unconstrained_1to1_beta1_summary
   This creates:
      summaries/unconstrained_1to1_beta1_summary_1.csv
      summaries/unconstrained_1to1_beta1_summary_2.csv
      ...

6. To save to an exact file:
      python summarize_results.py --results-file results/matching/unconstrained_1to1/run1.csv --save --out summaries/debug_summary.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def next_incremented_path(out_dir, prefix="summary", suffix=".csv"):
    """
    Return next available path:
        out_dir/prefix_1.csv
        out_dir/prefix_2.csv
        ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
    used = []

    for path in out_dir.glob(f"{prefix}_*{suffix}"):
        match = pattern.match(path.name)
        if match:
            used.append(int(match.group(1)))

    next_id = max(used, default=0) + 1
    return out_dir / f"{prefix}_{next_id}{suffix}"


def summarize_one(path, true_tau=1.0):
    df = pd.read_csv(path)

    out = {
        "file": Path(path).name,
        "path": str(path),
        "n_sim": len(df),
    }

    for col in ["method", "beta", "pstar", "n_mcmc", "dataset_file"]:
        if col in df.columns:
            vals = df[col].dropna().unique()
            out[col] = vals[0] if len(vals) == 1 else "mixed"

    if "est" in df.columns:
        out["mean_est"] = df["est"].mean()
        out["bias"] = df["est"].mean() - true_tau
        out["abs_bias"] = (df["est"] - true_tau).abs().mean()
        out["empirical_var_est"] = df["est"].var(ddof=1)

    for col in ["wiVar", "bwVar", "miVar", "wiVarclus", "miVarclus", "wiVarpaired", "miVarpaired"]:
        if col in df.columns:
            out[f"mean_{col}"] = df[col].mean()

    if {"ll", "ul"}.issubset(df.columns):
        out["coverage"] = ((df["ll"] <= true_tau) & (true_tau <= df["ul"])).mean()
        out["mean_ci_length"] = (df["ul"] - df["ll"]).mean()
    elif "covers_true" in df.columns:
        out["coverage"] = df["covers_true"].mean()

    if "acceptance_count" in df.columns and "n_mcmc" in df.columns:
        out["mean_acceptance_rate"] = (df["acceptance_count"] / df["n_mcmc"]).mean()
    elif "acceptance_count" in df.columns:
        out["mean_acceptance_count"] = df["acceptance_count"].mean()

    if "change_count" in df.columns and "n_sampling_draws" in df.columns:
        out["mean_change_rate"] = (df["change_count"] / df["n_sampling_draws"]).mean()
    elif "change_count" in df.columns:
        out["mean_change_count"] = df["change_count"].mean()

    if "changed_iteration_count" in df.columns and "n_mcmc" in df.columns:
        out["mean_changed_iteration_rate"] = (
            df["changed_iteration_count"] / df["n_mcmc"]
        ).mean()

    timing_cols = [
        "time_sim_total",
        "time_sim_prep",
        "time_method_call",
        "time_sim_post",
        "time_setup",
        "time_allocation",
        "time_mcmc_sampling",
        "time_mcmc_estimation",
        "time_mcmc_total",
        "time_summary",
        "time_method_total",
        "share_method_in_sim",
        "share_sampling_in_mcmc",
        "share_mcmc_in_method",
        "n_sampling_draws",
        "time_per_sampling_draw",
        "share_sampling_in_method",
        "share_estimation_in_method",
    ]

    for col in timing_cols:
        if col in df.columns:
            out[f"mean_{col}"] = df[col].mean()

    return out

def get_paths(results_file=None, results_dir=None, pattern="*.csv"):
    if results_file is not None:
        paths = [Path(results_file)]
    elif results_dir is not None:
        paths = sorted(Path(results_dir).glob(pattern))
    else:
        raise ValueError("Provide either --results-file or --results-dir.")

    if not paths:
        raise FileNotFoundError("No result CSV files found.")

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"These result files do not exist: {missing}")

    return paths


def print_compact(summary_df):
    # -----------------------------
    # Statistical summary (table)
    # -----------------------------
    stats_cols = [
        "file",
        "method",
        "beta",
        "n_sim",
        "mean_est",
        "bias",
        "abs_bias",
        "mean_miVar",
        "coverage",
        "mean_ci_length",
        "mean_acceptance_rate",
        "mean_change_rate",
        "mean_changed_iteration_rate",
    ]

    stats_cols = [c for c in stats_cols if c in summary_df.columns]

    stats = summary_df[stats_cols].copy()
    num_cols = stats.select_dtypes(include="number").columns
    stats[num_cols] = stats[num_cols].round(4)

    print("\n=== Statistical Summary ===")
    print(stats.to_string(index=False))

    # -----------------------------
    # Runtime summary (vertical)
    # -----------------------------
    runtime_cols = [
        "mean_time_sim_total",
        "mean_time_sim_prep",
        "mean_time_method_call",
        "mean_time_sim_post",
        "mean_share_method_in_sim",
        "mean_time_method_total",
        "mean_time_setup",
        "mean_time_allocation",
        "mean_time_summary",
        "mean_time_mcmc_total",
        "mean_time_mcmc_sampling",
        "mean_time_mcmc_estimation",
        "mean_n_sampling_draws",
        "mean_time_per_sampling_draw",
        "mean_share_sampling_in_mcmc",
        "mean_share_sampling_in_method",
        "mean_share_estimation_in_method",
        "mean_share_mcmc_in_method",
    ]

    runtime_cols = [c for c in runtime_cols if c in summary_df.columns]

    if runtime_cols:
        for _, row in summary_df.iterrows():

            print("\n" + "=" * 80)
            print(f"Runtime Profile: {row['file']}")
            print("=" * 80)

            print("\n[Simulation Loop]")
            for col in [
                "mean_time_sim_total",
                "mean_time_sim_prep",
                "mean_time_method_call",
                "mean_time_sim_post",
                "mean_share_method_in_sim",
            ]:
                if col in row and pd.notna(row[col]):
                    print(f"{col:<35} {row[col]:.6f}")

            print("\n[Method Internals]")
            for col in [
                "mean_time_method_total",
                "mean_time_setup",
                "mean_time_allocation",
                "mean_time_summary",
                "mean_time_mcmc_total",
                "mean_time_mcmc_sampling",
                "mean_time_mcmc_estimation",
            ]:
                if col in row and pd.notna(row[col]):
                    print(f"{col:<35} {row[col]:.6f}")

            print("\n[MCMC Breakdown]")
            for col in [
                "mean_n_sampling_draws",
                "mean_time_per_sampling_draw",
                "mean_share_sampling_in_mcmc",
                "mean_share_sampling_in_method",
                "mean_share_estimation_in_method",
                "mean_share_mcmc_in_method",
            ]:
                if col in row and pd.notna(row[col]):
                    if "draws" in col:
                        print(f"{col:<35} {int(row[col])}")
                    else:
                        print(f"{col:<35} {row[col]:.6f}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument("--true-tau", type=float, default=1.0)

    parser.add_argument("--save", action="store_true", help="Save summary CSV. Default is print only.")
    parser.add_argument("--out", type=str, default=None, help="Exact output path. Used only with --save.")
    parser.add_argument("--out-dir", type=str, default="summaries")
    parser.add_argument("--out-prefix", type=str, default="summary")

    args = parser.parse_args()

    paths = get_paths(args.results_file, args.results_dir, args.pattern)
    summary_df = pd.DataFrame([summarize_one(path, args.true_tau) for path in paths])

    print_compact(summary_df)

    if args.save:
        if args.out is not None:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = next_incremented_path(args.out_dir, prefix=args.out_prefix, suffix=".csv")

        summary_df.to_csv(out_path, index=False)
        print(f"\nFull summary saved to: {out_path}")
    else:
        print("\nSummary was not saved. Add --save to save it.")


if __name__ == "__main__":
    main()
