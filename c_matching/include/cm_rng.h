#ifndef CM_RNG_H
#define CM_RNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Small, deterministic PCG-XSH-RR generator.  Its output is reproducible on
 * every platform with 64-bit uint64_t and 32-bit uint32_t types.
 */
typedef struct CMRng {
    uint64_t state;
    uint64_t increment;
} CMRng;

/* Seed one independent stream.  Different stream values select PCG streams. */
void cm_rng_seed(CMRng *rng, uint64_t seed, uint64_t stream);

uint32_t cm_rng_u32(CMRng *rng);
uint64_t cm_rng_u64(CMRng *rng);

/* Uniform double in [0, 1), using 53 random bits. */
double cm_rng_uniform(CMRng *rng);

/* Unbiased integer in [0, bound).  bound must be nonzero. */
size_t cm_rng_bounded(CMRng *rng, size_t bound);

#ifdef __cplusplus
}
#endif

#endif
