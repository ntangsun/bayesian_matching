# Run unconstrained_1tok
# Run from project root:
#   .\scripts\run_unconstrained_1tok.ps1

python run_matching_from_datasets.py `
    --datasets datasets_sb01.npz `
    --method unconstrained_1tok `
    --n-mcmc 1000 `
    --beta 1 `
    --max-sim 100 `
    --out-dir results `
    --out-prefix unconstrained_1tok_beta1_profile `
    --profile

# python run_matching_from_datasets.py `
#     --datasets datasets_sb01.npz `
#     --method unconstrained_1tok `
#     --n-mcmc 1000 `
#     --beta 1 `
#     --out-dir results `
#     --out-prefix unconstrained_1tok_beta1
