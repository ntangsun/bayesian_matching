#include "cm_rng.h"

#include <limits.h>

uint32_t cm_rng_u32(CMRng *rng)
{
    const uint64_t old_state = rng->state;
    uint32_t xorshifted;
    uint32_t rotation;

    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    xorshifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    rotation = (uint32_t)(old_state >> 59u);
    return (xorshifted >> rotation) |
           (xorshifted << ((uint32_t)(-rotation) & 31u));
}

void cm_rng_seed(CMRng *rng, uint64_t seed, uint64_t stream)
{
    if (rng == NULL) {
        return;
    }

    rng->state = 0;
    rng->increment = (stream << 1u) | UINT64_C(1);
    (void)cm_rng_u32(rng);
    rng->state += seed;
    (void)cm_rng_u32(rng);
}

uint64_t cm_rng_u64(CMRng *rng)
{
    const uint64_t high = (uint64_t)cm_rng_u32(rng);
    const uint64_t low = (uint64_t)cm_rng_u32(rng);
    return (high << 32u) | low;
}

double cm_rng_uniform(CMRng *rng)
{
    /* 2^-53 is exactly representable as a binary floating-point value. */
    return (double)(cm_rng_u64(rng) >> 11u) * 0x1.0p-53;
}

size_t cm_rng_bounded(CMRng *rng, size_t bound)
{
    if (bound == 0) {
        return 0;
    }

#if SIZE_MAX <= UINT32_MAX
    {
        const uint32_t b = (uint32_t)bound;
        const uint32_t threshold = (uint32_t)(-b) % b;
        uint32_t value;
        do {
            value = cm_rng_u32(rng);
        } while (value < threshold);
        return (size_t)(value % b);
    }
#else
    {
        const uint64_t b = (uint64_t)bound;
        const uint64_t threshold = (uint64_t)(-b) % b;
        uint64_t value;
        do {
            value = cm_rng_u64(rng);
        } while (value < threshold);
        return (size_t)(value % b);
    }
#endif
}
