# Run constrained_1to1_gibbs
# Run from project root:
#   .\scripts\run_constrained_1to1_gibbs.ps1

python run_matching_from_datasets.py `
    --datasets datasets_sb01.npz `
    --method constrained_1to1_gibbs `
    --n-mcmc 1000 `
    --beta 1 `
    --max-sim 100 `
    --out-dir results `
    --out-prefix constrained_1to1_gibbs_beta1_profile `
    --profile

# python run_matching_from_datasets.py `
#     --datasets datasets_sb01.npz `
#     --method constrained_1to1_gibbs `
#     --n-mcmc 1000 `
#     --beta 1 `
#     --out-dir results `
#     --out-prefix constrained_1to1_gibbs_beta1
