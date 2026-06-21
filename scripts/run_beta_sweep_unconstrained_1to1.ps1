# Run beta sweep for unconstrained 1-to-1 matching.
# Run from project root:
#   .\scripts\run_beta_sweep_unconstrained_1to1.ps1

$betas = @(0, 1, 2, 3, 4, 5, 10, 20, 30, 40)

foreach ($b in $betas) {
    Write-Host "Running beta = $b"
    python run_matching_from_datasets_v2.py `
        --datasets datasets_sb01.npz `
        --method unconstrained_1to1 `
        --n-mcmc 1000 `
        --beta $b `
        --out-dir results `
        --out-prefix unconstrained_1to1_beta$b
}
