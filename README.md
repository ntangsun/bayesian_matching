# Probabilistic Matching for Causal Inference

This research project studies how uncertainty in the matching process affects treatment-effect estimation. Instead of treating one matched dataset as the final answer, I use probabilistic matching methods to sample many plausible matchings and evaluate both their statistical quality and computational cost.

## Research questions

- How do constrained and unconstrained matching designs affect bias, variance, covariate balance, and confidence-interval coverage?
- How does 1-to-1 matching compare with 1-to-K matching when several comparison units can be assigned to each recipient?
- How does the distance penalty, controlled by the regularization parameter `beta`, change the trade-off between match quality and matching uncertainty?
- Which sampling algorithm (Metropolis-Hastings, Gibbs sampling, or block Metropolis-Hastings) offers the best balance between convergence, state-space exploration, and runtime?

## Experimental approach

| Experiment | Purpose |
| --- | --- |
| Simulation studies | Measure bias, variance, and coverage when the true treatment effect is known. |
| Regularization paths | Study how estimates and covariate balance change across values of `beta`. |
| Matching-design comparisons | Compare constrained/unconstrained and 1-to-1/1-to-K matching. |
| Sampler comparisons | Compare MH, Gibbs, direct Monte Carlo, and experimental pure block-MH kernels. |
| Convergence diagnostics | Evaluate trace plots, running means, autocorrelation, split R-hat, effective sample size, and Monte Carlo error. |
| Runtime profiling | Compare sampling cost across algorithms and between Python and C implementations. |

A representative convergence experiment uses 300 recipients, 3,000 potential donors, and 1-to-5 constrained matching across four independently initialized chains. Gibbs sampling serves as the current baseline. The experimental block-MH kernels are evaluated separately at several block sizes to determine whether larger moves improve exploration enough to justify their computational cost.

## Methods and implementation

The project includes reproducible data generation, matching algorithms, treatment-effect estimation, MCMC diagnostics, simulation orchestration, and runtime profiling. The main implementation is in Python using NumPy, pandas, and Matplotlib. A separate C implementation is used to investigate performance improvements for large simulation studies.

This work demonstrates experience with:

- Monte Carlo and Markov chain Monte Carlo methods
- causal-inference matching and statistical simulation
- convergence diagnostics and uncertainty quantification
- algorithm design, profiling, and performance optimization
- reproducible research workflows in Python and C

## Repository guide

- `generate_datasets.py` creates simulation datasets.
- `run_matching_from_datasets.py` runs the main matching experiments.
- `matching_core.py` contains the matching states and sampling kernels.
- `run_mcmc_diagnostics.py` runs multi-chain convergence experiments.
- `mcmc_diagnostics.py` computes diagnostic statistics and plots.
- `experiments/` contains focused simulation and comparison studies.
- `c_matching/` contains the performance-oriented C implementation.

## Status

This is ongoing research. Current work focuses on determining whether new MH transition kernels can improve mixing while remaining faster per iteration than the Gibbs baseline. Final conclusions will be based on both computational efficiency and convergence, not runtime alone.
