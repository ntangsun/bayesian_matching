#!/usr/bin/env python3
"""
generate_datasets.py

Generate and store all causal-inference simulation datasets used by the
Bayesian matching scripts.

This separates data generation from matching, so the matching scripts can
reuse the exact same datasets without recreating them each run.

Example:
    python generate_datasets.py --n-sim 1000 --n-pop 100 --perc-rc 0.3 --sb 0.1 --out datasets_sb01.npz
"""

import argparse
import numpy as np


def create_causal_data(n_pop: int, perc_rc: float, sb: float, seed: int):
    """
    Python translation of createCausI / createCausI_highSB.

    g = 1 for recipients / treated
    g = 0 for donors / controls
    """
    rng = np.random.default_rng(seed)

    n_rc = int(perc_rc * n_pop)
    n_dn = n_pop - n_rc

    g = np.concatenate([np.ones(n_rc, dtype=int), np.zeros(n_dn, dtype=int)])

    num_match_var = 4
    var_rc = np.ones(num_match_var)

    mean_dn = np.zeros(num_match_var)
    mean_rc = sb * np.sqrt((1 + var_rc**2) / 2)

    sig_rc = np.eye(num_match_var)
    sig_dn = np.eye(num_match_var)

    x_rc = rng.multivariate_normal(mean_rc, sig_rc, size=n_rc)
    x_dn = rng.multivariate_normal(mean_dn, sig_dn, size=n_dn)
    X = np.vstack([x_rc, x_dn])

    X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eps = rng.normal(0, 1, size=n_pop)
    Y = 18 + g + 0.1 * X1 + 0.2 * X2 + 0.2 * X3 + 0.1 * X4 + eps

    return X, Y, g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--n-pop", type=int, default=100)
    parser.add_argument("--perc-rc", type=float, default=0.3)
    parser.add_argument("--sb", type=float, default=0.1, help="0.1 corresponds to createCausI; 0.3 to createCausI_highSB")
    parser.add_argument("--out", type=str, default="datasets.npz")
    args = parser.parse_args()

    X_list = []
    Y_list = []
    g_list = []
    seeds = []

    # R code uses set.seed(i) with i = 1,...,N.sim.
    # We use the same integer sequence as dataset seeds.
    for i in range(1, args.n_sim + 1):
        X, Y, g = create_causal_data(args.n_pop, args.perc_rc, args.sb, seed=i)
        X_list.append(X)
        Y_list.append(Y)
        g_list.append(g)
        seeds.append(i)

    np.savez_compressed(
        args.out,
        X=np.asarray(X_list),
        Y=np.asarray(Y_list),
        g=np.asarray(g_list),
        seeds=np.asarray(seeds),
        n_sim=args.n_sim,
        n_pop=args.n_pop,
        perc_rc=args.perc_rc,
        sb=args.sb,
    )

    print(f"Saved {args.n_sim} datasets to {args.out}")


if __name__ == "__main__":
    main()
