# C matching benchmark

This folder is a standalone C17 port of the reusable-dataset matching
pipeline. It does not modify or call the Python implementation.

Implemented methods:

- `unconstrained_1to1`
- `unconstrained_1tok`
- `unconstrained_1tok_mc`
- `constrained_1to1`
- `constrained_1to1_gibbs`
- `constrained_1tok`

`constrained_1tok_gibbs` is intentionally excluded for now.

## Build

The build requires a complete C17 toolchain. On Windows, the tested toolchain
is WinLibs MinGW-w64 GCC. From the repository root:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\c_matching\scripts\build.ps1 -Configuration Release
~~~

The build creates:

- `c_matching/bin/generate_datasets.exe`
- `c_matching/bin/run_matching_from_datasets.exe`
- `c_matching/bin/summarize_results.exe`

The implementation has no Python dependency and vendors no binary library.
The NPZ reader supports the Deflate-compressed NumPy archives already in the
repository; the generator writes standard NumPy-compatible, uncompressed NPZ.

## Equivalent profiled run

~~~powershell
.\c_matching\bin\run_matching_from_datasets.exe `
    --datasets datasets_sb01.npz `
    --method constrained_1to1 `
    --n-mcmc 1000 `
    --beta 1 `
    --max-sim 100 `
    --out-dir results_c `
    --out-prefix constrained_1to1_beta1_profile `
    --profile
~~~

The result CSV uses the same statistical and runtime column names as the
current Python runner. Dataset loading and CSV writing remain outside the
per-simulation timings.

## Run every active method

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\c_matching\scripts\run_all_methods.ps1
~~~

This builds Release mode, runs all six methods sequentially with the standard
profile settings, writes into a unique run directory under `results_c/`, and
invokes the C summarizer only on that run. Relative paths passed to the script
are interpreted from the repository root.

The C runner deliberately corrects the Python working-tree 1-to-K dispatch
issue: `--k` is honored by both unconstrained and constrained 1-to-K methods,
and the selected value is written to the result CSV.

## Reproducibility

The C implementation uses a documented PCG32 stream. It is deterministic
across supported platforms but intentionally does not emulate NumPy's PCG64
and SeedSequence implementation bit-for-bit. For fair runtime comparisons,
both runners should read the same existing NPZ dataset.
