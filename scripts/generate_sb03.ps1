# Generate reusable datasets, SB = 0.3
# Run from project root:
#   .\scripts\generate_sb03.ps1

python generate_datasets.py `
    --n-sim 1000 `
    --n-pop 100 `
    --perc-rc 0.3 `
    --sb 0.3 `
    --out datasets_sb03.npz
