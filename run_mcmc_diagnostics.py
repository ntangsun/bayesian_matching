#!/usr/bin/env python3
"""
run_mcmc_diagnostics.py

Sampler-only convergence diagnostics for constrained Bayesian matching.

What this runner does:
1. Generates or loads a reusable diagnostic dataset using n_r, K, rho, and seed.
2. Runs multiple constrained 1-to-K local sampler chains.
3. Records scalar summaries over time:
      tau(C_t), D(C_t), log_target(C_t), balance_l2(C_t)
4. Records time on three useful scales:
      raw proposal index, sweep = proposal / (n_r*K), sampler seconds
5. Saves trace CSVs, summary CSVs, and diagnostic plots.

Sweep convention:
- One iteration is one sampler call, regardless of how many slots that call
  proposes to update.
- Every method uses n_r*K raw proposals per sweep. Block size never rescales
  the proposal count.
- --thin is also measured in raw proposals.

Output layout:
    results/diagnostics/<data-and-target-setting>/<sampler-run>/
        traces/trace.csv
        summaries/summary.csv
        summaries/chain_summary.csv
        plots/*.png

What this runner intentionally does not do:
- It does not call run_matching_from_datasets.py.
- It does not fit regressions or compute variance estimates inside each MCMC
  iteration. Those are useful for simulation studies, but they would contaminate
  sampler runtime diagnostics.

First setting from our discussion:
    n_r = 300, K = 5, beta = 5, rho = 2, thin = 1

Short example:
    python run_mcmc_diagnostics.py --n-r 300 --k 5 --beta 5 --rho 2 --thin 1

Method choices:
- constrained_1tok: local MH sampler using matched-donor swaps and
  matched-vs-unmatched replacements.
- constrained_1tok_block: pure mixed-pool block MH sampler. One proposal
  jointly updates --block-size distinct recipient slots using the same number
  of currently unmatched donors.
- constrained_1tok_gibbs: local Gibbs sampler. One update chooses one
  recipient-slot and resamples from {current donor} plus unmatched donors.

rho warning:
- By default, the runner warns and asks before proceeding when
  rho <= --rho-warning-cutoff, whose default is 1.05.
- This matters because rho = 1 has no unmatched donors; local Gibbs updates
  become degenerate and local MH updates may mix slowly.
- Use --yes to proceed through the warning without prompting.

Long MH example:
    python run_mcmc_diagnostics.py `
      --method constrained_1tok `
      --data-model linear `
      --n-r 300 `
      --k 5 `
      --beta 5 `
      --rho 2 `
      --n-chains 4 `
      --n-sweeps 5000 `
      --thin 50 `
      --seed 1 `
      --out-dir results/diagnostics `
      --run-label mh_sweeps5000_thin50 `
      --print-every-sweeps 500

Long Gibbs example:
    python run_mcmc_diagnostics.py `
      --method constrained_1tok_gibbs `
      --data-model linear `
      --n-r 300 `
      --k 5 `
      --beta 5 `
      --rho 2 `
      --n-chains 4 `
      --n-sweeps 1000 `
      --thin 50 `
      --seed 1 `
      --sb 0.1 `
      --true-tau 1 `
      --data-dir diagnostic_datasets `
      --out-dir results/diagnostics `
      --max-acf-lag 200 `
      --print-every-sweeps 100

Pure block-MH experiment (four separate runs, one per block size):
    python run_mcmc_diagnostics.py `
      --method constrained_1tok_block `
      --block-size-grid 2,3,5,10 `
      --data-model linear `
      --n-r 300 `
      --k 5 `
      --beta 5 `
      --rho 2 `
      --n-chains 4 `
      --n-sweeps 1000 `
      --thin 50 `
      --seed 1 `
      --sb 0.1 `
      --true-tau 1 `
      --data-dir diagnostic_datasets `
      --out-dir results/diagnostics `
      --max-acf-lag 200 `
      --print-every-sweeps 100
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostic_data import format_float_for_filename, get_or_create_dataset
from matching_core import (
    distance_matrix_from_groups,
    gibbs_step_constrained_1tok,
    initial_matching,
    make_constrained_state,
    mh_step_constrained_1tok,
    mh_step_constrained_1tok_block,
    split_recipient_donor_arrays,
    state_balance_l2,
    state_fraction_initial_retained,
    state_tau,
)
from mcmc_diagnostics import make_standard_plots, summarize_chains, summarize_trace


DEFAULT_STARTS = ("greedy", "random", "high_distance", "perturbed_greedy")
METHODS = (
    "constrained_1tok",
    "constrained_1tok_block",
    "constrained_1tok_gibbs",
)


def parse_csv_values(raw: str | None, caster):
    """Parse comma-separated grid arguments such as '100,300'."""
    if raw is None or raw == "":
        return None
    return [caster(part.strip()) for part in raw.split(",") if part.strip()]


def values_from_single_or_grid(single, grid_raw: str | None, caster):
    """Use --x-grid when present; otherwise use the single --x value."""
    grid = parse_csv_values(grid_raw, caster)
    if grid is not None:
        return grid
    return [caster(single)]


def confirm_close_rho_values(
    rho_values: list[float],
    cutoff: float,
    assume_yes: bool,
) -> None:
    """
    Warn before running settings with rho close to 1.

    rho = 1 is the tightest feasible donor-supply case. Local Gibbs updates are
    degenerate there because no unmatched donors are available, and local MH
    updates can also mix slowly. This prompt is intentionally in the CLI layer
    so the lower-level sampler code remains non-interactive.
    """
    close_values = sorted({float(rho) for rho in rho_values if float(rho) <= cutoff})
    if not close_values:
        return

    print("")
    print("WARNING: rho is close to 1.")
    print(f"  cutoff: rho <= {cutoff:g}")
    print(f"  values: {', '.join(f'{rho:g}' for rho in close_values)}")
    print("  rho = 1 means the donor supply is tight: n_d = K * n_r.")
    print("  For constrained 1-to-K Gibbs, rho = 1 is degenerate because")
    print("  there are no unmatched donors available for a local Gibbs update.")
    print("")

    if assume_yes:
        print("Proceeding because --yes was provided.")
        return

    answer = input("Proceed with these rho values? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted before running diagnostics.")


def setting_slug(
    method: str,
    data_model: str,
    n_r: int,
    k: int,
    rho: float,
    beta: float,
    seed: int,
    n_sweeps: float,
    thin: int,
    run_label: str,
    block_size: int | None,
) -> str:
    """Filename-safe setting identifier used for traces, summaries, and plots."""
    rho_tag = format_float_for_filename(rho)
    beta_tag = format_float_for_filename(beta)
    sweeps_tag = format_float_for_filename(n_sweeps)
    method_tag = (
        f"{method}_b{block_size}"
        if method == "constrained_1tok_block"
        else method
    )
    label = f"_{run_label}" if run_label else ""
    return (
        f"{method_tag}_{data_model}_nr{n_r}_k{k}_rho{rho_tag}_"
        f"beta{beta_tag}_seed{seed}_sweeps{sweeps_tag}_thin{thin}{label}"
    )


def setting_group_slug(
    data_model: str,
    n_r: int,
    k: int,
    rho: float,
    beta: float,
    seed: int,
) -> str:
    """Shared folder name for runs using the same data and target setting."""
    rho_tag = format_float_for_filename(rho)
    beta_tag = format_float_for_filename(beta)
    return (
        f"{data_model}_nr{n_r}_k{k}_rho{rho_tag}_"
        f"beta{beta_tag}_seed{seed}"
    )


def sampler_run_slug(
    method: str,
    n_sweeps: float,
    thin: int,
    run_label: str,
    block_size: int | None,
) -> str:
    """Folder name for one sampler run inside a shared setting folder."""
    sweeps_tag = format_float_for_filename(n_sweeps)
    method_tag = (
        f"{method}_b{block_size}"
        if method == "constrained_1tok_block"
        else method
    )
    label = f"_{run_label}" if run_label else ""
    return f"{method_tag}_sweeps{sweeps_tag}_thin{thin}{label}"


def unique_setting_dir(out_dir: Path, slug: str) -> Path:
    """
    Return a run-specific output folder without overwriting older runs.

    First run:
        results/diagnostics/<setting>/<slug>/

    If that already exists:
        results/diagnostics/<setting>/<slug>_run2/
        results/diagnostics/<setting>/<slug>_run3/

    This keeps every setting run self-contained and preserves previous results.
    """
    base = out_dir / slug
    if not base.exists():
        return base

    run_id = 2
    while True:
        candidate = out_dir / f"{slug}_run{run_id}"
        if not candidate.exists():
            return candidate
        run_id += 1


def choose_start_kind(chain_index: int, start_kinds: tuple[str, ...]) -> str:
    """
    Pick dispersed starts for the first chains, then random starts after that.

    This lets --n-chains exceed the named start list without failing.
    """
    if chain_index < len(start_kinds):
        return start_kinds[chain_index]
    return "random"


def append_trace_row(
    rows: list[dict[str, object]],
    *,
    setting_id: str,
    method: str,
    data_model: str,
    dataset_file: Path,
    n_r: int,
    n_d: int,
    k: int,
    rho: float,
    beta: float,
    pstar: float,
    block_size: int | None,
    chain: int,
    init: str,
    proposal: int,
    n_proposals_per_sweep: int,
    sampler_seconds: float,
    wall_seconds: float,
    accepted: bool,
    move_type: str,
    accepted_count: int,
    state,
    y_rc_mean: float,
    x_rc_mean: np.ndarray,
) -> None:
    """
    Save one trace row.

    With thin=1 this runs every proposal. The state object keeps cached summary
    values so this row is cheap to create.
    """
    sweep = proposal / n_proposals_per_sweep
    hamming = state.match_cols.size - state.retained_count
    rows.append(
        {
            "setting_id": setting_id,
            "method": method,
            "data_model": data_model,
            "dataset_file": str(dataset_file),
            "n_r": n_r,
            "n_d": n_d,
            "k": k,
            "rho": rho,
            "beta": beta,
            "pstar": pstar,
            "block_size": block_size,
            "chain": chain,
            "init": init,
            "proposal": proposal,
            "sweep": sweep,
            "sampler_seconds": sampler_seconds,
            "wall_seconds": wall_seconds,
            "accepted": int(accepted),
            "move_type": move_type,
            "accepted_count": accepted_count,
            "running_acceptance": accepted_count / proposal if proposal > 0 else 0.0,
            "tau": state_tau(state, y_rc_mean),
            "d_total": state.d_total,
            "log_target": -beta * state.d_total,
            "balance_l2": state_balance_l2(state, x_rc_mean),
            "hamming_from_initial": hamming,
            "fraction_initial_retained": state_fraction_initial_retained(state),
        }
    )


def run_one_setting(
    *,
    method: str,
    data_model: str,
    n_r: int,
    k: int,
    rho: float,
    beta: float,
    pstar: float,
    block_size: int | None,
    n_chains: int,
    n_sweeps: float,
    thin: int,
    seed: int,
    sb: float,
    true_tau: float,
    data_dir: Path,
    out_dir: Path,
    start_kinds: tuple[str, ...],
    max_acf_lag: int,
    make_plots: bool,
    run_label: str,
    print_every_sweeps: float,
) -> tuple[Path, Path, Path]:
    """
    Run one method/data/parameter setting.

    Supported methods:
    - constrained_1tok: local MH swap/replacement sampler.
    - constrained_1tok_block: pure mixed-pool block MH sampler.
    - constrained_1tok_gibbs: local Gibbs slot update using unmatched donors.
    """
    if method not in METHODS:
        known = ", ".join(METHODS)
        raise ValueError(f"Unknown method={method!r}. Known methods: {known}")

    if thin < 1:
        raise ValueError("--thin must be at least 1.")
    if n_sweeps <= 0:
        raise ValueError("--n-sweeps must be positive.")
    if not 0 <= pstar <= 1:
        raise ValueError("--pstar must be between 0 and 1.")
    if method == "constrained_1tok_block":
        if block_size is None or block_size < 1:
            raise ValueError("--block-size must be at least 1 for the block sampler.")
    elif block_size is not None:
        raise ValueError("--block-size is only valid with constrained_1tok_block.")

    dataset_file, X, Y, g, metadata = get_or_create_dataset(
        data_dir=data_dir,
        data_model=data_model,
        n_r=n_r,
        k=k,
        rho=rho,
        seed=seed,
        sb=sb,
        true_tau=true_tau,
    )

    X_rc, X_dn, Y_rc, Y_dn = split_recipient_donor_arrays(X, Y, g)
    n_d = X_dn.shape[0]
    dist = distance_matrix_from_groups(X_rc, X_dn, dist_type=2)

    if method == "constrained_1tok_block":
        n_unmatched = n_d - n_r * k
        if block_size > n_r:
            raise ValueError("--block-size cannot exceed --n-r.")
        if block_size > n_unmatched:
            raise ValueError(
                f"--block-size={block_size} exceeds the unmatched-pool size "
                f"({n_unmatched}) for this setting."
            )

    n_proposals_per_sweep = n_r * k
    n_proposals = int(np.ceil(float(n_sweeps) * n_proposals_per_sweep))

    slug = setting_slug(
        method=method,
        data_model=data_model,
        n_r=n_r,
        k=k,
        rho=rho,
        beta=beta,
        seed=seed,
        n_sweeps=n_sweeps,
        thin=thin,
        run_label=run_label,
        block_size=block_size,
    )

    setting_root = out_dir / setting_group_slug(
        data_model=data_model,
        n_r=n_r,
        k=k,
        rho=rho,
        beta=beta,
        seed=seed,
    )
    run_slug = sampler_run_slug(
        method=method,
        n_sweeps=n_sweeps,
        thin=thin,
        run_label=run_label,
        block_size=block_size,
    )
    setting_dir = unique_setting_dir(setting_root, run_slug)
    trace_dir = setting_dir / "traces"
    summary_dir = setting_dir / "summaries"
    plot_dir = setting_dir / "plots"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    trace_rows: list[dict[str, object]] = []
    y_rc_mean = float(Y_rc.mean())
    x_rc_mean = X_rc.mean(axis=0)

    for chain in range(n_chains):
        init = choose_start_kind(chain, start_kinds)
        rng = np.random.default_rng(seed + 10_000 * (chain + 1))
        init_match = initial_matching(init, dist=dist, k=k, rng=rng)
        state = make_constrained_state(init_match, dist=dist, X_dn=X_dn, Y_dn=Y_dn)

        sampler_seconds = 0.0
        accepted_count = 0
        wall_start = time.perf_counter()
        next_progress_sweep = print_every_sweeps if print_every_sweeps > 0 else None

        append_trace_row(
            trace_rows,
            setting_id=slug,
            method=method,
            data_model=data_model,
            dataset_file=dataset_file,
            n_r=n_r,
            n_d=n_d,
            k=k,
            rho=rho,
            beta=beta,
            pstar=pstar,
            block_size=block_size,
            chain=chain + 1,
            init=init,
            proposal=0,
            n_proposals_per_sweep=n_proposals_per_sweep,
            sampler_seconds=sampler_seconds,
            wall_seconds=0.0,
            accepted=False,
            move_type="initial",
            accepted_count=accepted_count,
            state=state,
            y_rc_mean=y_rc_mean,
            x_rc_mean=x_rc_mean,
        )

        for proposal in range(1, n_proposals + 1):
            proposal_start = time.perf_counter()
            if method == "constrained_1tok":
                accepted, move_type = mh_step_constrained_1tok(
                    state=state,
                    dist=dist,
                    X_dn=X_dn,
                    Y_dn=Y_dn,
                    beta=beta,
                    pstar=pstar,
                    rng=rng,
                )
            elif method == "constrained_1tok_block":
                accepted, move_type = mh_step_constrained_1tok_block(
                    state=state,
                    dist=dist,
                    X_dn=X_dn,
                    Y_dn=Y_dn,
                    beta=beta,
                    block_size=block_size,
                    rng=rng,
                )
            elif method == "constrained_1tok_gibbs":
                accepted, move_type = gibbs_step_constrained_1tok(
                    state=state,
                    dist=dist,
                    X_dn=X_dn,
                    Y_dn=Y_dn,
                    beta=beta,
                    rng=rng,
                )
            else:
                raise RuntimeError(f"Unhandled method={method!r}")
            sampler_seconds += time.perf_counter() - proposal_start
            accepted_count += int(accepted)

            if next_progress_sweep is not None:
                current_sweep = proposal / n_proposals_per_sweep
                if current_sweep >= next_progress_sweep or proposal == n_proposals:
                    print(
                        "  "
                        f"chain {chain + 1}/{n_chains} ({init}): "
                        f"sweep {current_sweep:.1f}/{n_sweeps:g}, "
                        f"changed/accepted {accepted_count}/{proposal} "
                        f"({accepted_count / proposal:.3f})"
                    )
                    while next_progress_sweep <= current_sweep:
                        next_progress_sweep += print_every_sweeps

            if proposal % thin == 0 or proposal == n_proposals:
                append_trace_row(
                    trace_rows,
                    setting_id=slug,
                    method=method,
                    data_model=data_model,
                    dataset_file=dataset_file,
                    n_r=n_r,
                    n_d=n_d,
                    k=k,
                    rho=rho,
                    beta=beta,
                    pstar=pstar,
                    block_size=block_size,
                    chain=chain + 1,
                    init=init,
                    proposal=proposal,
                    n_proposals_per_sweep=n_proposals_per_sweep,
                    sampler_seconds=sampler_seconds,
                    wall_seconds=time.perf_counter() - wall_start,
                    accepted=accepted,
                    move_type=move_type,
                    accepted_count=accepted_count,
                    state=state,
                    y_rc_mean=y_rc_mean,
                    x_rc_mean=x_rc_mean,
                )

    trace_df = pd.DataFrame(trace_rows)
    trace_path = trace_dir / "trace.csv"
    trace_df.to_csv(trace_path, index=False)

    summary_df = summarize_trace(trace_df, max_lag=max_acf_lag)
    for key, value in {
        "setting_id": slug,
        "setting_dir": str(setting_dir),
        "method": method,
        "data_model": data_model,
        "dataset_file": str(dataset_file),
        "dataset_created": bool(metadata["created"]),
        "n_r": n_r,
        "n_d": n_d,
        "k": k,
        "rho": rho,
        "beta": beta,
        "pstar": pstar,
        "block_size": block_size,
        "n_chains": n_chains,
        "n_sweeps": n_sweeps,
        "n_proposals": n_proposals,
        "thin": thin,
        "seed": seed,
        "sb": sb,
        "true_tau": true_tau,
    }.items():
        summary_df[key] = value

    summary_path = summary_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    chain_df = summarize_chains(trace_df)
    for key, value in {
        "setting_id": slug,
        "setting_dir": str(setting_dir),
        "method": method,
        "data_model": data_model,
        "dataset_file": str(dataset_file),
        "n_r": n_r,
        "n_d": n_d,
        "k": k,
        "rho": rho,
        "beta": beta,
        "pstar": pstar,
        "block_size": block_size,
        "n_sweeps": n_sweeps,
        "thin": thin,
        "seed": seed,
    }.items():
        chain_df[key] = value

    chain_summary_path = summary_dir / "chain_summary.csv"
    chain_df.to_csv(chain_summary_path, index=False)

    if make_plots:
        make_standard_plots(
            trace_df,
            plots_dir=plot_dir,
            slug="",
            max_lag=max_acf_lag,
        )

    print(f"Saved setting folder: {setting_dir}")
    print(f"Saved trace: {trace_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved chain summary: {chain_summary_path}")

    return trace_path, summary_path, chain_summary_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="constrained_1tok", choices=METHODS)
    parser.add_argument("--data-model", type=str, default="linear")

    parser.add_argument("--n-r", type=int, default=300)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=2.0)

    parser.add_argument("--n-r-grid", type=str, default=None, help="Comma list, e.g. 100,300")
    parser.add_argument("--k-grid", type=str, default=None, help="Comma list, e.g. 1,5,10")
    parser.add_argument("--beta-grid", type=str, default=None, help="Comma list, e.g. 5,20")
    parser.add_argument("--rho-grid", type=str, default=None, help="Comma list, e.g. 1.0,1.1")
    parser.add_argument(
        "--rho-warning-cutoff",
        type=float,
        default=1.05,
        help="Warn and ask before running rho values at or below this cutoff.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Proceed through interactive warnings without prompting.",
    )

    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument(
        "--n-sweeps",
        type=float,
        default=20.0,
        help="Sweeps to run; one sweep is n_r*K raw proposals for every method.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Save one trace row every this many raw proposals.",
    )
    parser.add_argument("--pstar", type=float, default=0.5)
    parser.add_argument(
        "--block-size",
        type=int,
        default=2,
        help="Block size for constrained_1tok_block (default: 2).",
    )
    parser.add_argument(
        "--block-size-grid",
        type=str,
        default=None,
        help="Comma-separated block sizes, e.g. 2,3,5,10.",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sb", type=float, default=0.1)
    parser.add_argument("--true-tau", type=float, default=1.0)

    parser.add_argument("--data-dir", type=str, default="diagnostic_datasets")
    parser.add_argument("--out-dir", type=str, default="results/diagnostics")
    parser.add_argument("--start-kinds", type=str, default=",".join(DEFAULT_STARTS))
    parser.add_argument("--max-acf-lag", type=int, default=200)
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument(
        "--print-every-sweeps",
        type=float,
        default=10.0,
        help="Print chain progress every this many sweeps. Use 0 to disable.",
    )
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    n_r_values = values_from_single_or_grid(args.n_r, args.n_r_grid, int)
    k_values = values_from_single_or_grid(args.k, args.k_grid, int)
    beta_values = values_from_single_or_grid(args.beta, args.beta_grid, float)
    rho_values = values_from_single_or_grid(args.rho, args.rho_grid, float)
    if args.method == "constrained_1tok_block":
        block_size_values = values_from_single_or_grid(
            args.block_size,
            args.block_size_grid,
            int,
        )
    else:
        if args.block_size_grid is not None:
            raise ValueError(
                "--block-size-grid is only valid with constrained_1tok_block."
            )
        block_size_values = [None]
    start_kinds = tuple(x.strip() for x in args.start_kinds.split(",") if x.strip())

    if args.n_chains < 1:
        raise ValueError("--n-chains must be at least 1.")
    if args.rho_warning_cutoff >= 1:
        confirm_close_rho_values(
            rho_values=rho_values,
            cutoff=args.rho_warning_cutoff,
            assume_yes=args.yes,
        )

    settings = list(
        itertools.product(
            n_r_values,
            k_values,
            beta_values,
            rho_values,
            block_size_values,
        )
    )
    print(f"Running {len(settings)} diagnostic setting(s).")

    written_summaries = []
    for n_r, k, beta, rho, block_size in settings:
        block_text = f", block_size={block_size}" if block_size is not None else ""
        print(
            "Setting: "
            f"n_r={n_r}, K={k}, beta={beta:g}, rho={rho:g}, "
            f"chains={args.n_chains}, sweeps={args.n_sweeps:g}, "
            f"thin={args.thin}{block_text}"
        )
        _, summary_path, _ = run_one_setting(
            method=args.method,
            data_model=args.data_model,
            n_r=n_r,
            k=k,
            rho=rho,
            beta=beta,
            pstar=args.pstar,
            block_size=block_size,
            n_chains=args.n_chains,
            n_sweeps=args.n_sweeps,
            thin=args.thin,
            seed=args.seed,
            sb=args.sb,
            true_tau=args.true_tau,
            data_dir=Path(args.data_dir),
            out_dir=Path(args.out_dir),
            start_kinds=start_kinds,
            max_acf_lag=args.max_acf_lag,
            make_plots=not args.no_plots,
            run_label=args.run_label,
            print_every_sweeps=args.print_every_sweeps,
        )
        written_summaries.append(summary_path)

    if len(written_summaries) > 1:
        combined = pd.concat(
            [pd.read_csv(path) for path in written_summaries],
            ignore_index=True,
        )
        setting_roots = {path.parents[2] for path in written_summaries}
        if len(setting_roots) == 1:
            combined_path = setting_roots.pop() / "combined_latest_summary.csv"
        else:
            combined_path = (
                Path(args.out_dir)
                / "combined_summaries"
                / "combined_latest_summary.csv"
            )
            combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
