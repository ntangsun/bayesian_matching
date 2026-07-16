#!/usr/bin/env python3
"""
mcmc_diagnostics.py

Numerical and graphical diagnostics for saved MCMC traces.

What this file does:
- Computes autocorrelation, integrated autocorrelation time (IACT), effective
  sample size (ESS), ESS/sec, Monte Carlo standard error (MCSE), and split Rhat.
- Makes trace, running-mean, and autocorrelation plots.

What this file does not do:
- It does not run the sampler.
- It does not generate data.
- It does not fit treatment-effect regressions.

This separation keeps the sampler runtime clean: the runner records trace
values, then this file analyzes those saved values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_VALUE_COLS = ("tau", "d_total", "log_target", "balance_l2")


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    FFT autocorrelation for one scalar chain.

    The result starts at lag 0, so acf[0] is 1.0 unless the chain is constant.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))

    x = x - x.mean()
    var0 = float(np.dot(x, x))
    if var0 == 0:
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 1.0
        return out

    fft_len = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=fft_len)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=fft_len)[:n]
    acf = acov[: max_lag + 1] / acov[0]
    acf[0] = 1.0
    return acf


def iact_from_acf(acf: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time using a simple positive sequence.

    We stop adding lags when the autocorrelation first becomes non-positive.
    This is conservative enough for a first diagnostic pass and easy to read.
    """
    if len(acf) == 0:
        return float("nan")

    total = 1.0
    for rho in acf[1:]:
        if not np.isfinite(rho) or rho <= 0:
            break
        total += 2.0 * float(rho)
    return max(total, 1.0)


def ess_mcse(values: np.ndarray, max_lag: int | None = None) -> dict[str, float]:
    """Return IACT, ESS, and MCSE for one scalar chain."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": float(n), "iact": np.nan, "ess": np.nan, "mcse": np.nan}

    acf = autocorrelation(x, max_lag=max_lag)
    iact = iact_from_acf(acf)
    ess = n / iact
    sd = float(np.std(x, ddof=1))
    mcse = sd / np.sqrt(ess) if ess > 0 else np.nan
    return {"n": float(n), "iact": iact, "ess": ess, "mcse": mcse}


def split_rhat(chain_values: Iterable[np.ndarray]) -> float:
    """
    Basic split Rhat across chains.

    Each chain is split in half, then the usual between/within variance ratio
    is computed. This is not rank-normalized Rhat, but it is transparent and
    sufficient for the first implementation.
    """
    chains = [np.asarray(v, dtype=float) for v in chain_values]
    chains = [v[np.isfinite(v)] for v in chains if len(v) >= 4]
    if len(chains) < 2:
        return float("nan")

    min_len = min(len(v) for v in chains)
    split_len = min_len // 2
    if split_len < 2:
        return float("nan")

    split_chains = []
    for values in chains:
        trimmed = values[: 2 * split_len]
        split_chains.append(trimmed[:split_len])
        split_chains.append(trimmed[split_len:])

    arr = np.vstack(split_chains)
    m, n = arr.shape
    chain_means = arr.mean(axis=1)
    chain_vars = arr.var(axis=1, ddof=1)

    W = float(chain_vars.mean())
    if W == 0:
        return float("nan")

    B = float(n * chain_means.var(ddof=1))
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def summarize_trace(
    trace_df: pd.DataFrame,
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
    max_lag: int = 200,
) -> pd.DataFrame:
    """
    Build one summary row per monitored scalar.

    ESS/sec uses sampler_seconds, which is accumulated only around MH proposal
    updates. This avoids counting plotting or CSV-writing time as sampler time.
    """
    rows: list[dict[str, float | str]] = []
    grouped = list(trace_df.groupby("chain", sort=True))

    for col in value_cols:
        chain_arrays = [group[col].to_numpy(dtype=float) for _, group in grouped]
        rhat = split_rhat(chain_arrays)

        per_chain = []
        for chain_id, group in grouped:
            stats = ess_mcse(group[col].to_numpy(dtype=float), max_lag=max_lag)
            sampler_seconds = float(group["sampler_seconds"].max())
            ess_per_sec = stats["ess"] / sampler_seconds if sampler_seconds > 0 else np.nan
            per_chain.append(
                {
                    "chain": int(chain_id),
                    "ess": stats["ess"],
                    "ess_per_sampler_sec": ess_per_sec,
                    "iact": stats["iact"],
                    "mcse": stats["mcse"],
                    "mean": float(group[col].mean()),
                    "sd": float(group[col].std(ddof=1)),
                }
            )

        per_chain_df = pd.DataFrame(per_chain)
        rows.append(
            {
                "variable": col,
                "rhat_split": rhat,
                "mean_ess": float(per_chain_df["ess"].mean()),
                "min_ess": float(per_chain_df["ess"].min()),
                "mean_ess_per_sampler_sec": float(
                    per_chain_df["ess_per_sampler_sec"].mean()
                ),
                "mean_iact": float(per_chain_df["iact"].mean()),
                "mean_mcse": float(per_chain_df["mcse"].mean()),
                "mean_value": float(per_chain_df["mean"].mean()),
                "mean_chain_sd": float(per_chain_df["sd"].mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_chains(trace_df: pd.DataFrame) -> pd.DataFrame:
    """One operational summary row per chain."""
    rows = []
    for chain_id, group in trace_df.groupby("chain", sort=True):
        final = group.iloc[-1]
        n_proposals = int(group["proposal"].max())
        sampler_seconds = float(group["sampler_seconds"].max())
        rows.append(
            {
                "chain": int(chain_id),
                "init": str(final["init"]),
                "n_saved": int(len(group)),
                "n_proposals": n_proposals,
                "final_sweep": float(final["sweep"]),
                "sampler_seconds": sampler_seconds,
                "wall_seconds": float(group["wall_seconds"].max()),
                "change_or_acceptance_rate": float(final["running_acceptance"]),
                "acceptance_rate": float(final["running_acceptance"]),
                "time_per_proposal": (
                    sampler_seconds / n_proposals if n_proposals > 0 else np.nan
                ),
                "time_per_sweep": (
                    sampler_seconds / final["sweep"] if final["sweep"] > 0 else np.nan
                ),
                "final_tau": float(final["tau"]),
                "final_d_total": float(final["d_total"]),
                "final_log_target": float(final["log_target"]),
                "final_balance_l2": float(final["balance_l2"]),
                "final_fraction_initial_retained": float(
                    final["fraction_initial_retained"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _finish_plot(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_trace(
    trace_df: pd.DataFrame,
    value_col: str,
    x_col: str,
    out_path: str | Path,
    title: str,
) -> None:
    """Line trace for one scalar summary across all chains."""
    plt.figure(figsize=(10, 5))
    for chain_id, group in trace_df.groupby("chain", sort=True):
        plt.plot(group[x_col], group[value_col], linewidth=0.8, alpha=0.85, label=f"chain {chain_id}")
    plt.xlabel(x_col)
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    _finish_plot(Path(out_path))


def plot_running_mean(
    trace_df: pd.DataFrame,
    value_col: str,
    x_col: str,
    out_path: str | Path,
    title: str,
) -> None:
    """Running mean plot for Monte Carlo estimator stabilization."""
    plt.figure(figsize=(10, 5))
    for chain_id, group in trace_df.groupby("chain", sort=True):
        values = group[value_col].to_numpy(dtype=float)
        running = np.cumsum(values) / np.arange(1, len(values) + 1)
        plt.plot(group[x_col], running, linewidth=0.9, alpha=0.9, label=f"chain {chain_id}")
    plt.xlabel(x_col)
    plt.ylabel(f"running mean {value_col}")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    _finish_plot(Path(out_path))


def plot_acf(
    trace_df: pd.DataFrame,
    value_col: str,
    out_path: str | Path,
    title: str,
    max_lag: int = 200,
) -> None:
    """Autocorrelation plot for one scalar summary across chains."""
    plt.figure(figsize=(10, 5))
    for chain_id, group in trace_df.groupby("chain", sort=True):
        acf = autocorrelation(group[value_col].to_numpy(dtype=float), max_lag=max_lag)
        plt.plot(np.arange(len(acf)), acf, linewidth=0.9, alpha=0.9, label=f"chain {chain_id}")
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.xlabel("lag in saved samples")
    plt.ylabel("autocorrelation")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    _finish_plot(Path(out_path))


def make_standard_plots(
    trace_df: pd.DataFrame,
    plots_dir: str | Path,
    slug: str = "",
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
    max_lag: int = 200,
) -> list[Path]:
    """
    Create the standard diagnostic plot set.

    Because thin may be 1, proposal plots are fine-grained. Sweep and seconds
    plots use the same saved points with different x-axis interpretations.
    """
    plots_dir = Path(plots_dir)
    written: list[Path] = []
    prefix = f"{slug}_" if slug else ""

    for value_col in value_cols:
        for x_col in ("proposal", "sweep", "sampler_seconds"):
            out = plots_dir / f"{prefix}trace_{value_col}_by_{x_col}.png"
            plot_trace(
                trace_df,
                value_col=value_col,
                x_col=x_col,
                out_path=out,
                title=f"Trace of {value_col} by {x_col}",
            )
            written.append(out)

        out = plots_dir / f"{prefix}running_mean_{value_col}_by_proposal.png"
        plot_running_mean(
            trace_df,
            value_col=value_col,
            x_col="proposal",
            out_path=out,
            title=f"Running mean of {value_col}",
        )
        written.append(out)

        out = plots_dir / f"{prefix}acf_{value_col}.png"
        plot_acf(
            trace_df,
            value_col=value_col,
            out_path=out,
            title=f"ACF of {value_col}",
            max_lag=max_lag,
        )
        written.append(out)

    return written
