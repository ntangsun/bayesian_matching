#ifndef CM_DATASET_H
#define CM_DATASET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Arrays use NumPy C-order (row-major) layout:
 *
 *   X[(simulation * n_pop + unit) * n_cov + covariate]
 *   Y[ simulation * n_pop + unit]
 *   g[ simulation * n_pop + unit]
 *   seeds[simulation]
 */
typedef struct DatasetCollection {
    size_t n_sim;
    size_t n_pop;
    size_t n_cov;
    double *X;
    double *Y;
    int *g;
    uint64_t *seeds;
    double perc_rc;
    double sb;
} DatasetCollection;

/* Set every member to zero/NULL. Safe to call before cm_dataset_free(). */
void cm_dataset_init(DatasetCollection *dataset);

/*
 * Allocate X, Y, g, and seeds for the supplied dimensions. The collection
 * must be initialized (or previously freed). All allocations are zeroed.
 */
int cm_dataset_alloc(DatasetCollection *dataset,
                     size_t n_sim,
                     size_t n_pop,
                     size_t n_cov,
                     char *error,
                     size_t error_size);

/* Release all owned arrays and reset the collection. NULL is accepted. */
void cm_dataset_free(DatasetCollection *dataset);

/*
 * Load the eight arrays written by generate_datasets.py:
 * X, Y, g, seeds, n_sim, n_pop, perc_rc, and sb.
 *
 * The reader supports ZIP stored/deflated entries and NumPy NPY v1.0,
 * little-endian C-order float64/int32/int64 arrays. On failure, out is left
 * empty and a human-readable message is written to error when provided.
 */
int cm_dataset_load_npz(const char *path,
                        DatasetCollection *out,
                        char *error,
                        size_t error_size);

/*
 * Save the same eight members as a NumPy-compatible NPZ archive. ZIP entries
 * are stored (not compressed), deliberately avoiding an external zlib
 * dependency. The resulting archive is readable by numpy.load().
 */
int cm_dataset_save_npz(const char *path,
                        const DatasetCollection *dataset,
                        char *error,
                        size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* CM_DATASET_H */
