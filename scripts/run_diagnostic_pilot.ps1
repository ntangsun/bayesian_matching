# Run the first sampler-only MCMC diagnostic setting.
#
# This script is intentionally separate from the old run_*.ps1 scripts.
# The old scripts measure full simulation work, including treatment-effect
# estimation inside each iteration. This diagnostic measures the matching
# sampler itself and records trace values for convergence checks.
#
# First setting from the discussion:
#   n_r  = 300 recipients
#   K    = 5 donors per recipient
#   beta = 5
#   rho  = 2 donor-supply ratio, so n_d = rho * K * n_r = 3000
#
# thin = 1 means every proposal is saved to the trace and used in plots.
#
# Run from project root:
#   .\scripts\run_diagnostic_pilot.ps1

python run_mcmc_diagnostics.py `
    --method constrained_1tok `
    --data-model linear `
    --n-r 300 `
    --k 5 `
    --beta 5 `
    --rho 2 `
    --n-chains 4 `
    --n-sweeps 20 `
    --thin 1 `
    --seed 1 `
    --out-dir results/diagnostics
