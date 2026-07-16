# Run unconstrained_1tok_mc
# Run from project root:
#   .\scripts\run_unconstrained_1tok_mc.ps1

python run_matching_from_datasets.py `
    --datasets datasets_sb01.npz `
    --method unconstrained_1tok_mc `
    --n-mcmc 1000 `
    --beta 1 `
    --max-sim 100 `
    --out-dir results/matching/unconstrained_1tok_mc `
    --out-prefix unconstrained_1tok_mc_beta1_profile `
    --profile

# python run_matching_from_datasets.py `
#     --datasets datasets_sb01.npz `
#     --method unconstrained_1tok_mc `
#     --n-mcmc 1000 `
#     --beta 1 `
#     --out-dir results/matching/unconstrained_1tok_mc `
#     --out-prefix unconstrained_1tok_mc_beta1
