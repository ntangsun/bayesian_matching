# Run unconstrained_1to1
# Run from project root:
#   .\scripts\run_unconstrained_1to1.ps1

python run_matching_from_datasets.py `
    --datasets datasets_sb01.npz `
    --method unconstrained_1to1 `
    --n-mcmc 1000 `
    --beta 1 `
    --max-sim 100 `
    --out-dir results/matching/unconstrained_1to1 `
    --out-prefix unconstrained_1to1_beta1_profile `
    --profile

# python run_matching_from_datasets.py `
#     --datasets datasets_sb01.npz `
#     --method unconstrained_1to1 `
#     --n-mcmc 1000 `
#     --beta 1 `
#     --out-dir results/matching/unconstrained_1to1 `
#     --out-prefix unconstrained_1to1_beta1 `
#     --profile 
