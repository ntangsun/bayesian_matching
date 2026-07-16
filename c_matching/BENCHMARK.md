# Validation benchmark

Validated on 2026-07-14 with:

- Input: `datasets_sb01.npz`
- Replications: 100
- Sampling iterations: 1,000
- `beta=1`, `pstar=0.5`, `k=2`
- Profiling enabled
- C build: GCC 16.1.0, C17, `-O3 -march=native`

The table compares mean `time_sim_total` from the existing Python profile
CSVs with the C validation run. Both implementations exclude dataset loading
and CSV writing from this field.

| Method | Python seconds/sim | C seconds/sim | Speedup |
|---|---:|---:|---:|
| `unconstrained_1to1` | 0.073021717 | 0.001198354 | 60.9x |
| `unconstrained_1tok` | 0.278062075 | 0.001949997 | 142.6x |
| `unconstrained_1tok_mc` | 1.198834200 | 0.007057486 | 169.9x |
| `constrained_1to1` | 0.326686890 | 0.000453778 | 719.9x |
| `constrained_1to1_gibbs` | 0.432986751 | 0.002183557 | 198.3x |
| `constrained_1tok` | 0.322556635 | 0.000535000 | 602.9x |

For the requested `constrained_1to1` case, the sampling phase was about
444.4x faster and the estimation phase about 1,799.3x faster. The complete
six-method C driver, including six NPZ loads, six process launches, CSV
writes, and combined summarization, completed in about 2.20 seconds.

## Statistical comparison

The C implementation uses PCG32 rather than NumPy PCG64, so individual chains
are not expected to match row-for-row. Aggregate estimates remained close:

| Method | Python mean estimate | C mean estimate |
|---|---:|---:|
| `unconstrained_1to1` | 1.043335 | 1.043596 |
| `unconstrained_1tok` | 1.043740 | 1.042393 |
| `unconstrained_1tok_mc` | 1.044341 | 1.044237 |
| `constrained_1to1` | 1.039551 | 1.036458 |
| `constrained_1to1_gibbs` | 1.037963 | 1.038040 |
| `constrained_1tok` | 1.042440 | 1.044348 |

The largest difference in these six mean estimates was approximately 0.0031.

## Validation performed

- Warning-free Release build of all three executables.
- Exact Deflate reconstruction of the existing `X.npy` payload.
- Successful loading of both 1,000- and 10,000-replication Python NPZ files.
- C-generated NPZ files load successfully with NumPy.
- Repeated generation with identical arguments is byte-for-byte deterministic.
- All six methods completed 100 profiled replications.
- All numeric outputs were finite.
- CI coverage, timer identities, draw counts, simulation numbering, and
  method-specific `k`/acceptance fields passed automated invariant checks.
- The C and Python summarizers both parsed the generated C result CSV.

Python reference files used were the latest available profile for each method:
`unconstrained_1to1_beta1_profile_2.csv`,
`unconstrained_1tok_beta1_profile_3.csv`,
`unconstrained_1tok_mc_beta1_profile_1.csv`,
`constrained_1to1_beta1_profile_3.csv`,
`constrained_1to1_gibbs_beta1_profile_1.csv`, and
`constrained_1tok_beta1_profile_2.csv`.
