#!/usr/bin/env python3
"""
diagnostic_data.py

Dataset generation and caching for MCMC convergence diagnostics.

Why this file exists:
- The old simulation generator is parameterized by total population size and
  treated percentage. The convergence study is parameterized by n_r, K, and
  rho, where rho controls donor supply.
- For constrained 1-to-K matching we need n_d >= K * n_r. We therefore build
  diagnostic datasets directly from:

      n_d = round(rho * K * n_r)

- Datasets are saved and reused. Running the same setting with the same seed
  should load the same X, Y, and g instead of regenerating new random data.

Extending later:
- Add a new generator function, for example generate_nonlinear_data(...).
- Register it in DATA_MODELS.
- The diagnostic runner can then use --data-model nonlinear without changing
  the sampler code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def format_float_for_filename(value: float) -> str:
    """Make values like 1.1 filename-safe: 1p1 instead of 1.1."""
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def diagnostic_dataset_path(
    out_dir: str | Path,
    data_model: str,
    n_r: int,
    k: int,
    rho: float,
    seed: int,
    sb: float,
) -> Path:
    """
    Return the deterministic dataset path for this data-generating setting.

    beta is intentionally not part of the filename. beta changes the MCMC
    target distribution over matchings, not the generated covariates/outcomes.
    Reusing the same dataset across beta values gives cleaner comparisons.
    """
    rho_tag = format_float_for_filename(rho)
    sb_tag = format_float_for_filename(sb)
    filename = (
        f"{data_model}_nr{n_r}_k{k}_rho{rho_tag}_"
        f"sb{sb_tag}_seed{seed}.npz"
    )
    return Path(out_dir) / filename


def compute_n_d(n_r: int, k: int, rho: float) -> int:
    """
    Convert the donor-supply ratio into an integer donor count.

    rho = 1.0 means the tightest feasible case: every donor is used.
    rho > 1.0 means there are unmatched donors available.
    """
    n_d = int(round(float(rho) * int(k) * int(n_r)))
    min_required = int(k) * int(n_r)
    if n_d < min_required:
        raise ValueError(
            f"rho={rho} gives n_d={n_d}, but constrained 1-to-K needs "
            f"at least K*n_r={min_required} donors."
        )
    return n_d


def generate_linear_data(
    n_r: int,
    n_d: int,
    sb: float,
    seed: int,
    true_tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the same basic linear causal data model used by the old scripts.

    g = 1 for recipients / treated units.
    g = 0 for donors / control units.

    The outcome model has a true treatment effect true_tau. The diagnostic
    sampler uses Y only to monitor tau(C_t), not to fit regressions.
    """
    rng = np.random.default_rng(seed)

    g = np.concatenate([np.ones(n_r, dtype=int), np.zeros(n_d, dtype=int)])

    num_match_var = 4
    var_rc = np.ones(num_match_var)
    mean_dn = np.zeros(num_match_var)
    mean_rc = sb * np.sqrt((1 + var_rc**2) / 2)

    x_rc = rng.multivariate_normal(mean_rc, np.eye(num_match_var), size=n_r)
    x_dn = rng.multivariate_normal(mean_dn, np.eye(num_match_var), size=n_d)
    X = np.vstack([x_rc, x_dn])

    X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eps = rng.normal(0, 1, size=n_r + n_d)
    Y = 18 + true_tau * g + 0.1 * X1 + 0.2 * X2 + 0.2 * X3 + 0.1 * X4 + eps

    return X, Y, g


DATA_MODELS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    "linear": generate_linear_data,
}


def get_or_create_dataset(
    data_dir: str | Path,
    data_model: str,
    n_r: int,
    k: int,
    rho: float,
    seed: int,
    sb: float = 0.1,
    true_tau: float = 1.0,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    Load an existing diagnostic dataset or generate and save it if missing.

    Returns:
        path, X, Y, g, metadata

    The metadata is also stored inside the .npz so a future reader can recover
    exactly how the dataset was generated.
    """
    if data_model not in DATA_MODELS:
        known = ", ".join(sorted(DATA_MODELS))
        raise ValueError(f"Unknown data_model={data_model!r}. Known models: {known}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    n_d = compute_n_d(n_r, k, rho)
    path = diagnostic_dataset_path(data_dir, data_model, n_r, k, rho, seed, sb)

    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        X = loaded["X"]
        Y = loaded["Y"]
        g = loaded["g"]
        metadata = {
            "data_model": str(loaded["data_model"]),
            "n_r": int(loaded["n_r"]),
            "n_d": int(loaded["n_d"]),
            "k": int(loaded["k"]),
            "rho": float(loaded["rho"]),
            "sb": float(loaded["sb"]),
            "seed": int(loaded["seed"]),
            "true_tau": float(loaded["true_tau"]),
            "created": False,
        }
        return path, X, Y, g, metadata

    generator = DATA_MODELS[data_model]
    X, Y, g = generator(n_r=n_r, n_d=n_d, sb=sb, seed=seed, true_tau=true_tau)

    np.savez_compressed(
        path,
        X=X,
        Y=Y,
        g=g,
        data_model=np.array(data_model),
        n_r=np.array(n_r),
        n_d=np.array(n_d),
        k=np.array(k),
        rho=np.array(rho),
        sb=np.array(sb),
        seed=np.array(seed),
        true_tau=np.array(true_tau),
    )

    metadata = {
        "data_model": data_model,
        "n_r": n_r,
        "n_d": n_d,
        "k": k,
        "rho": rho,
        "sb": sb,
        "seed": seed,
        "true_tau": true_tau,
        "created": True,
    }
    return path, X, Y, g, metadata
