# Run constrained_1tok_gibbs
# Run from project root:
#   .\scripts\run_constrained_1tok_gibbs.ps1

python run_matching_from_datasets.py `
    --datasets datasets_sb01.npz `
    --method constrained_1tok_gibbs `
    --n-mcmc 1000 `
    --beta 1 `
    --k 2 `
    --max-sim 100 `
    --out-dir results/matching/constrained_1tok_gibbs `
    --out-prefix constrained_1tok_gibbs_beta1_profile `
    --profile

# python run_matching_from_datasets.py `
#     --datasets datasets_sb01.npz `
#     --method constrained_1tok_gibbs `
#     --n-mcmc 1000 `
#     --beta 1 `
#     --k 2 `
#     --out-dir results/matching/constrained_1tok_gibbs `
#     --out-prefix constrained_1tok_gibbs_beta1
