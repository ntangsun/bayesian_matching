#include "cm_dataset.h"
#include "cm_rng.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct NormalGenerator {
    CMRng rng;
    int has_spare;
    double spare;
} NormalGenerator;

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "Usage: %s [--n-sim N] [--n-pop N] [--perc-rc X] [--sb X] [--out FILE]\n"
            "Defaults: --n-sim 1000 --n-pop 100 --perc-rc 0.3 --sb 0.1 --out datasets.npz\n",
            program);
}

static int parse_size(const char *text, size_t *out) {
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || *text == '\0' || *text == '-') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int parse_double_value(const char *text, double *out) {
    char *end = NULL;
    double value;
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

static double normal_draw(NormalGenerator *generator) {
    const double two_pi = 6.283185307179586476925286766559;
    double u1;
    double u2;
    double radius;
    double angle;
    if (generator->has_spare) {
        generator->has_spare = 0;
        return generator->spare;
    }
    u1 = 1.0 - cm_rng_uniform(&generator->rng);
    u2 = cm_rng_uniform(&generator->rng);
    radius = sqrt(-2.0 * log(u1));
    angle = two_pi * u2;
    generator->spare = radius * sin(angle);
    generator->has_spare = 1;
    return radius * cos(angle);
}

int main(int argc, char **argv) {
    size_t n_sim = 1000;
    size_t n_pop = 100;
    double perc_rc = 0.3;
    double sb = 0.1;
    const char *out_path = "datasets.npz";
    DatasetCollection dataset;
    char error[512];
    size_t s;
    int i;

    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *value;
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(stdout, argv[0]);
            return EXIT_SUCCESS;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value after %s\n", arg);
            return EXIT_FAILURE;
        }
        value = argv[++i];
        if (strcmp(arg, "--n-sim") == 0) {
            if (!parse_size(value, &n_sim) || n_sim == 0) {
                fprintf(stderr, "--n-sim must be positive\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(arg, "--n-pop") == 0) {
            if (!parse_size(value, &n_pop) || n_pop == 0) {
                fprintf(stderr, "--n-pop must be positive\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(arg, "--perc-rc") == 0) {
            if (!parse_double_value(value, &perc_rc)) {
                fprintf(stderr, "Invalid --perc-rc value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(arg, "--sb") == 0) {
            if (!parse_double_value(value, &sb)) {
                fprintf(stderr, "Invalid --sb value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(arg, "--out") == 0) {
            out_path = value;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            usage(stderr, argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (!(perc_rc > 0.0 && perc_rc < 1.0)) {
        fprintf(stderr, "--perc-rc must be strictly between 0 and 1\n");
        return EXIT_FAILURE;
    }

    cm_dataset_init(&dataset);
    if (!cm_dataset_alloc(&dataset, n_sim, n_pop, 4, error, sizeof(error))) {
        fprintf(stderr, "Allocation failed: %s\n", error);
        return EXIT_FAILURE;
    }
    dataset.perc_rc = perc_rc;
    dataset.sb = sb;

    for (s = 0; s < n_sim; ++s) {
        NormalGenerator normal;
        size_t n_rc = (size_t)(perc_rc * (double)n_pop);
        size_t unit;
        size_t cov;
        uint64_t seed = (uint64_t)s + UINT64_C(1);
        memset(&normal, 0, sizeof(normal));
        cm_rng_seed(&normal.rng, seed, UINT64_C(54));
        dataset.seeds[s] = seed;

        for (unit = 0; unit < n_pop; ++unit) {
            size_t row = s * n_pop + unit;
            int recipient = unit < n_rc;
            dataset.g[row] = recipient;
            for (cov = 0; cov < 4; ++cov) {
                dataset.X[row * 4 + cov] = (recipient ? sb : 0.0) + normal_draw(&normal);
            }
        }
        for (unit = 0; unit < n_pop; ++unit) {
            size_t row = s * n_pop + unit;
            const double *x = dataset.X + row * 4;
            dataset.Y[row] = 18.0 + (double)dataset.g[row]
                + 0.1 * x[0] + 0.2 * x[1] + 0.2 * x[2] + 0.1 * x[3]
                + normal_draw(&normal);
        }
    }

    if (!cm_dataset_save_npz(out_path, &dataset, error, sizeof(error))) {
        fprintf(stderr, "Save failed: %s\n", error);
        cm_dataset_free(&dataset);
        return EXIT_FAILURE;
    }
    printf("Saved %zu datasets to %s\n", n_sim, out_path);
    cm_dataset_free(&dataset);
    return EXIT_SUCCESS;
}
