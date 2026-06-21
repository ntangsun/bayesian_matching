# Generate reusable datasets, SB = 0.1
# Run from project root:
#   .\scripts\generate_sb01.ps1

python generate_datasets.py `
    --n-sim 1000 `
    --n-pop 100 `
    --perc-rc 0.3 `
    --sb 0.1 `
    --out datasets_sb01.npz
