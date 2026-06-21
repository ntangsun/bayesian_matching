# Run constrained_1to1
# Run from project root:
#   .\scripts\run_constrained_1to1.ps1

python run_matching_from_datasets.py `
    --datasets datasets_sb01.npz `
    --method constrained_1to1 `
    --n-mcmc 1000 `
    --beta 1 `
    --out-dir results `
    --out-prefix constrained_1to1_beta1
