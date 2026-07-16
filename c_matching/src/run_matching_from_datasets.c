#include "cm_dataset.h"
#include "cm_matching.h"
#include "cm_rng.h"
#include "cm_timer.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define CM_PATH_SEP '\\'
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#define CM_PATH_SEP '/'
#endif

#define CM_PATH_CAP 4096

typedef struct CliOptions {
    const char *datasets;
    CMMethod method;
    size_t n_mcmc;
    double beta;
    size_t k;
    double pstar;
    size_t max_sim;
    bool has_max_sim;
    double true_tau;
    bool profile;
    const char *out_dir;
    const char *out_prefix;
    const char *out;
    size_t print_every;
} CliOptions;

typedef struct OutputRow {
    CMMethodResult result;
    double ll;
    double ul;
    bool has_ci;
    int covers_true;
    double time_sim_prep;
    double time_method_call;
    double time_sim_post;
    double time_sim_total;
    double share_method_in_sim;
} OutputRow;

static void print_usage(FILE *stream, const char *program) {
    fprintf(stream,
            "Usage: %s --datasets FILE --method METHOD [options]\n"
            "\n"
            "Methods:\n"
            "  unconstrained_1to1\n"
            "  unconstrained_1tok\n"
            "  unconstrained_1tok_mc\n"
            "  constrained_1to1\n"
            "  constrained_1to1_gibbs\n"
            "  constrained_1tok\n"
            "\n"
            "Options:\n"
            "  --n-mcmc N       Sampling iterations (default: 100000)\n"
            "  --beta X         Distance regularization (default: 1)\n"
            "  --k N            Donors per recipient for 1-to-k (default: 2)\n"
            "  --pstar X        Constrained MH swap probability (default: 0.5)\n"
            "  --max-sim N      Run only the first N replications\n"
            "  --true-tau X     True effect used for coverage (default: 1)\n"
            "  --profile        Emit Python-compatible runtime columns\n"
            "  --out-dir DIR    Auto-numbered output directory (default: results_c)\n"
            "  --out-prefix P   Auto-numbered filename prefix (default: results)\n"
            "  --out FILE       Exact output path instead of auto-numbering\n"
            "  --print-every N  Progress interval (default: 10; 0 disables)\n"
            "  -h, --help       Show this help\n",
            program);
}

static int parse_size(const char *text, size_t *out) {
    char *end = NULL;
    uintmax_t value;
    if (text == NULL || *text == '\0' || *text == '-') return 0;
    errno = 0;
    value = strtoumax(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int parse_double_value(const char *text, double *out) {
    char *end = NULL;
    double value;
    if (text == NULL || *text == '\0') return 0;
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

static int require_value(int argc, char **argv, int *index, const char **value) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "Missing value after %s\n", argv[*index]);
        return 0;
    }
    ++(*index);
    *value = argv[*index];
    return 1;
}

static int parse_cli(int argc, char **argv, CliOptions *options) {
    int i;
    bool have_method = false;
    memset(options, 0, sizeof(*options));
    options->n_mcmc = 100000;
    options->beta = 1.0;
    options->k = 2;
    options->pstar = 0.5;
    options->true_tau = 1.0;
    options->out_dir = "results_c";
    options->out_prefix = "results";
    options->print_every = 10;

    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *value = NULL;
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(stdout, argv[0]);
            return 2;
        } else if (strcmp(arg, "--profile") == 0) {
            options->profile = true;
        } else if (strcmp(arg, "--datasets") == 0) {
            if (!require_value(argc, argv, &i, &options->datasets)) return 0;
        } else if (strcmp(arg, "--method") == 0) {
            if (!require_value(argc, argv, &i, &value)) return 0;
            if (!cm_method_parse(value, &options->method)) {
                fprintf(stderr, "Unknown method: %s\n", value);
                return 0;
            }
            have_method = true;
        } else if (strcmp(arg, "--n-mcmc") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_size(value, &options->n_mcmc) || options->n_mcmc < 2) {
                fprintf(stderr, "--n-mcmc must be an integer of at least 2\n");
                return 0;
            }
        } else if (strcmp(arg, "--beta") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_double_value(value, &options->beta)) {
                fprintf(stderr, "Invalid --beta value\n");
                return 0;
            }
        } else if (strcmp(arg, "--k") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_size(value, &options->k) || options->k == 0) {
                fprintf(stderr, "--k must be a positive integer\n");
                return 0;
            }
        } else if (strcmp(arg, "--pstar") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_double_value(value, &options->pstar) ||
                options->pstar < 0.0 || options->pstar > 1.0) {
                fprintf(stderr, "--pstar must be between 0 and 1\n");
                return 0;
            }
        } else if (strcmp(arg, "--max-sim") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_size(value, &options->max_sim)) {
                fprintf(stderr, "--max-sim must be a nonnegative integer\n");
                return 0;
            }
            options->has_max_sim = true;
        } else if (strcmp(arg, "--true-tau") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_double_value(value, &options->true_tau)) {
                fprintf(stderr, "Invalid --true-tau value\n");
                return 0;
            }
        } else if (strcmp(arg, "--out-dir") == 0) {
            if (!require_value(argc, argv, &i, &options->out_dir)) return 0;
        } else if (strcmp(arg, "--out-prefix") == 0) {
            if (!require_value(argc, argv, &i, &options->out_prefix)) return 0;
        } else if (strcmp(arg, "--out") == 0) {
            if (!require_value(argc, argv, &i, &options->out)) return 0;
        } else if (strcmp(arg, "--print-every") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_size(value, &options->print_every)) {
                fprintf(stderr, "--print-every must be a nonnegative integer\n");
                return 0;
            }
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            return 0;
        }
    }
    if (options->datasets == NULL || !have_method) {
        fprintf(stderr, "Both --datasets and --method are required.\n");
        return 0;
    }
    return 1;
}

static int make_one_directory(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 1;
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 1;
#endif
    return 0;
}

static int make_directories(const char *path) {
    char buffer[CM_PATH_CAP];
    size_t length;
    size_t i;
    if (path == NULL || *path == '\0') return 1;
    length = strlen(path);
    if (length >= sizeof(buffer)) return 0;
    memcpy(buffer, path, length + 1);
    while (length > 0 && (buffer[length - 1] == '/' || buffer[length - 1] == '\\'))
        buffer[--length] = '\0';
    if (length == 0) return 1;
    for (i = 1; i < length; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\\') {
            char saved = buffer[i];
            if (i == 2 && buffer[1] == ':') continue;
            buffer[i] = '\0';
            if (*buffer != '\0' && !make_one_directory(buffer)) return 0;
            buffer[i] = saved;
        }
    }
    return make_one_directory(buffer);
}

static int ensure_parent_directory(const char *path) {
    char buffer[CM_PATH_CAP];
    char *last_slash;
    char *last_backslash;
    size_t length = strlen(path);
    if (length >= sizeof(buffer)) return 0;
    memcpy(buffer, path, length + 1);
    last_slash = strrchr(buffer, '/');
    last_backslash = strrchr(buffer, '\\');
    if (last_backslash != NULL && (last_slash == NULL || last_backslash > last_slash))
        last_slash = last_backslash;
    if (last_slash == NULL) return 1;
    *last_slash = '\0';
    return make_directories(buffer);
}

static int parse_numbered_filename(const char *name, const char *prefix, uint64_t *number) {
    size_t prefix_length = strlen(prefix);
    const char *cursor;
    char *end = NULL;
    uintmax_t value;
    if (strncmp(name, prefix, prefix_length) != 0 || name[prefix_length] != '_') return 0;
    cursor = name + prefix_length + 1;
    if (!isdigit((unsigned char)*cursor)) return 0;
    errno = 0;
    value = strtoumax(cursor, &end, 10);
    if (errno != 0 || end == cursor || strcmp(end, ".csv") != 0 || value > UINT64_MAX) return 0;
    *number = (uint64_t)value;
    return 1;
}

static uint64_t largest_existing_suffix(const char *directory, const char *prefix) {
    uint64_t largest = 0;
#ifdef _WIN32
    char pattern[CM_PATH_CAP];
    WIN32_FIND_DATAA data;
    HANDLE handle;
    int written = snprintf(pattern, sizeof(pattern), "%s\\%s_*.csv", directory, prefix);
    if (written < 0 || (size_t)written >= sizeof(pattern)) return 0;
    handle = FindFirstFileA(pattern, &data);
    if (handle == INVALID_HANDLE_VALUE) return 0;
    do {
        uint64_t value;
        if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
            parse_numbered_filename(data.cFileName, prefix, &value) && value > largest)
            largest = value;
    } while (FindNextFileA(handle, &data));
    FindClose(handle);
#else
    DIR *dir = opendir(directory);
    struct dirent *entry;
    if (dir == NULL) return 0;
    while ((entry = readdir(dir)) != NULL) {
        uint64_t value;
        if (parse_numbered_filename(entry->d_name, prefix, &value) && value > largest)
            largest = value;
    }
    closedir(dir);
#endif
    return largest;
}

static int choose_output_path(const CliOptions *options, char *path, size_t capacity) {
    int written;
    if (options->out != NULL) {
        if (strlen(options->out) >= capacity || !ensure_parent_directory(options->out)) return 0;
        memcpy(path, options->out, strlen(options->out) + 1);
        return 1;
    }
    if (!make_directories(options->out_dir)) return 0;
    written = snprintf(path, capacity, "%s%c%s_%" PRIu64 ".csv",
                       options->out_dir, CM_PATH_SEP, options->out_prefix,
                       largest_existing_suffix(options->out_dir, options->out_prefix) + UINT64_C(1));
    return written >= 0 && (size_t)written < capacity;
}

static void csv_write_string(FILE *file, const char *value) {
    const char *cursor;
    bool quote = false;
    if (value == NULL) value = "";
    for (cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor == ',' || *cursor == '"' || *cursor == '\r' || *cursor == '\n') {
            quote = true;
            break;
        }
    }
    if (!quote) {
        fputs(value, file);
        return;
    }
    fputc('"', file);
    for (cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor == '"') fputc('"', file);
        fputc(*cursor, file);
    }
    fputc('"', file);
}

static const char *const RESULT_FIELDS[] = {
    "acceptance_count", "beta", "bwVar", "change_count",
    "changed_iteration_count", "covers_true", "dataset_file", "est",
    "est_med", "estdiff", "k", "ll", "method", "miVar",
    "miVarclus", "miVarpaired", "n_mcmc", "n_sampling_draws", "pstar",
    "result_file", "share_estimation_in_method", "share_mcmc_in_method",
    "share_method_in_sim", "share_sampling_in_mcmc", "share_sampling_in_method",
    "sim", "time_allocation", "time_mcmc_estimation", "time_mcmc_sampling",
    "time_mcmc_total", "time_method_call", "time_method_total",
    "time_per_sampling_draw", "time_setup", "time_sim_post", "time_sim_prep",
    "time_sim_total", "time_summary", "ul", "wiVar", "wiVarclus",
    "wiVarpaired"
};

static bool field_is_enabled(const char *field, uint32_t flags, bool profile,
                             bool include_ci) {
    if (strcmp(field, "covers_true") == 0 || strcmp(field, "ll") == 0 ||
        strcmp(field, "ul") == 0)
        return include_ci;
    if (strcmp(field, "acceptance_count") == 0)
        return (flags & CM_RESULT_HAS_ACCEPTANCE_COUNT) != 0;
    if (strcmp(field, "change_count") == 0)
        return (flags & CM_RESULT_HAS_CHANGE_COUNT) != 0;
    if (strcmp(field, "changed_iteration_count") == 0)
        return (flags & CM_RESULT_HAS_CHANGED_ITERATION_COUNT) != 0;
    if (strcmp(field, "estdiff") == 0)
        return (flags & CM_RESULT_HAS_ESTDIFF) != 0;
    if (strcmp(field, "k") == 0)
        return (flags & CM_RESULT_HAS_K) != 0;
    if (strcmp(field, "miVarclus") == 0 || strcmp(field, "wiVarclus") == 0)
        return (flags & CM_RESULT_HAS_CLUSTER_VARIANCE) != 0;
    if (strcmp(field, "miVarpaired") == 0 || strcmp(field, "wiVarpaired") == 0)
        return (flags & CM_RESULT_HAS_PAIRED_VARIANCE) != 0;
    if (strncmp(field, "time_", 5) == 0 || strncmp(field, "share_", 6) == 0 ||
        strcmp(field, "n_sampling_draws") == 0)
        return profile;
    return true;
}

static void write_field_value(FILE *file, const char *field, const OutputRow *row,
                              size_t sim, const CliOptions *options,
                              const char *result_path) {
    const CMMethodResult *r = &row->result;
    const CMMethodTiming *t = &r->timing;
    if (strcmp(field, "acceptance_count") == 0) fprintf(file, "%" PRIu64, r->acceptance_count);
    else if (strcmp(field, "beta") == 0) fprintf(file, "%.17g", options->beta);
    else if (strcmp(field, "bwVar") == 0) fprintf(file, "%.17g", r->bw_var);
    else if (strcmp(field, "change_count") == 0) fprintf(file, "%" PRIu64, r->change_count);
    else if (strcmp(field, "changed_iteration_count") == 0) fprintf(file, "%" PRIu64, r->changed_iteration_count);
    else if (strcmp(field, "covers_true") == 0) {
        if (row->has_ci) fprintf(file, "%d", row->covers_true);
    }
    else if (strcmp(field, "dataset_file") == 0) csv_write_string(file, options->datasets);
    else if (strcmp(field, "est") == 0) fprintf(file, "%.17g", r->est);
    else if (strcmp(field, "est_med") == 0) fprintf(file, "%.17g", r->est_med);
    else if (strcmp(field, "estdiff") == 0) fprintf(file, "%.17g", r->estdiff);
    else if (strcmp(field, "k") == 0) fprintf(file, "%zu", r->k);
    else if (strcmp(field, "ll") == 0) {
        if (row->has_ci) fprintf(file, "%.17g", row->ll);
    }
    else if (strcmp(field, "method") == 0) fputs(cm_method_name(r->method), file);
    else if (strcmp(field, "miVar") == 0) fprintf(file, "%.17g", r->mi_var);
    else if (strcmp(field, "miVarclus") == 0) fprintf(file, "%.17g", r->mi_var_clus);
    else if (strcmp(field, "miVarpaired") == 0) fprintf(file, "%.17g", r->mi_var_paired);
    else if (strcmp(field, "n_mcmc") == 0) fprintf(file, "%zu", options->n_mcmc);
    else if (strcmp(field, "n_sampling_draws") == 0) fprintf(file, "%" PRIu64, t->n_sampling_draws);
    else if (strcmp(field, "pstar") == 0) fprintf(file, "%.17g", options->pstar);
    else if (strcmp(field, "result_file") == 0) csv_write_string(file, result_path);
    else if (strcmp(field, "share_estimation_in_method") == 0) fprintf(file, "%.17g", t->share_estimation_in_method);
    else if (strcmp(field, "share_mcmc_in_method") == 0) fprintf(file, "%.17g", t->share_mcmc_in_method);
    else if (strcmp(field, "share_method_in_sim") == 0) fprintf(file, "%.17g", row->share_method_in_sim);
    else if (strcmp(field, "share_sampling_in_mcmc") == 0) fprintf(file, "%.17g", t->share_sampling_in_mcmc);
    else if (strcmp(field, "share_sampling_in_method") == 0) fprintf(file, "%.17g", t->share_sampling_in_method);
    else if (strcmp(field, "sim") == 0) fprintf(file, "%zu", sim + 1);
    else if (strcmp(field, "time_allocation") == 0) fprintf(file, "%.17g", t->allocation);
    else if (strcmp(field, "time_mcmc_estimation") == 0) fprintf(file, "%.17g", t->mcmc_estimation);
    else if (strcmp(field, "time_mcmc_sampling") == 0) fprintf(file, "%.17g", t->mcmc_sampling);
    else if (strcmp(field, "time_mcmc_total") == 0) fprintf(file, "%.17g", t->mcmc_total);
    else if (strcmp(field, "time_method_call") == 0) fprintf(file, "%.17g", row->time_method_call);
    else if (strcmp(field, "time_method_total") == 0) fprintf(file, "%.17g", t->method_total);
    else if (strcmp(field, "time_per_sampling_draw") == 0) fprintf(file, "%.17g", t->time_per_sampling_draw);
    else if (strcmp(field, "time_setup") == 0) fprintf(file, "%.17g", t->setup);
    else if (strcmp(field, "time_sim_post") == 0) fprintf(file, "%.17g", row->time_sim_post);
    else if (strcmp(field, "time_sim_prep") == 0) fprintf(file, "%.17g", row->time_sim_prep);
    else if (strcmp(field, "time_sim_total") == 0) fprintf(file, "%.17g", row->time_sim_total);
    else if (strcmp(field, "time_summary") == 0) fprintf(file, "%.17g", t->summary);
    else if (strcmp(field, "ul") == 0) {
        if (row->has_ci) fprintf(file, "%.17g", row->ul);
    }
    else if (strcmp(field, "wiVar") == 0) fprintf(file, "%.17g", r->wi_var);
    else if (strcmp(field, "wiVarclus") == 0) fprintf(file, "%.17g", r->wi_var_clus);
    else if (strcmp(field, "wiVarpaired") == 0) fprintf(file, "%.17g", r->wi_var_paired);
}

static int write_results_csv(const char *path, const OutputRow *rows, size_t n_rows,
                             const CliOptions *options) {
    FILE *file;
    size_t i;
    size_t j;
    uint32_t flags;
    bool first;
    bool include_ci = false;
    if (n_rows == 0) return 0;
    file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Could not open %s for writing: %s\n", path, strerror(errno));
        return 0;
    }
    flags = rows[0].result.flags;
    for (i = 0; i < n_rows; ++i) {
        if (rows[i].has_ci) {
            include_ci = true;
            break;
        }
    }
    first = true;
    for (j = 0; j < sizeof(RESULT_FIELDS) / sizeof(RESULT_FIELDS[0]); ++j) {
        if (!field_is_enabled(RESULT_FIELDS[j], flags, options->profile, include_ci)) continue;
        if (!first) fputc(',', file);
        fputs(RESULT_FIELDS[j], file);
        first = false;
    }
    fputs("\r\n", file);
    for (i = 0; i < n_rows; ++i) {
        first = true;
        for (j = 0; j < sizeof(RESULT_FIELDS) / sizeof(RESULT_FIELDS[0]); ++j) {
            if (!field_is_enabled(RESULT_FIELDS[j], flags, options->profile, include_ci)) continue;
            if (!first) fputc(',', file);
            write_field_value(file, RESULT_FIELDS[j], &rows[i], i, options, path);
            first = false;
        }
        fputs("\r\n", file);
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Failed while closing %s\n", path);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    CliOptions cli;
    DatasetCollection datasets;
    OutputRow *rows = NULL;
    CMRunOptions run_options;
    char error[512];
    char output_path[CM_PATH_CAP];
    size_t n_sim;
    size_t s;
    int parse_status = parse_cli(argc, argv, &cli);
    int exit_code = EXIT_FAILURE;

    if (parse_status == 2) return EXIT_SUCCESS;
    if (parse_status == 0) {
        print_usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }
    if (!choose_output_path(&cli, output_path, sizeof(output_path))) {
        fprintf(stderr, "Could not create or choose the output path.\n");
        return EXIT_FAILURE;
    }

    cm_dataset_init(&datasets);
    if (!cm_dataset_load_npz(cli.datasets, &datasets, error, sizeof(error))) {
        fprintf(stderr, "Dataset load failed: %s\n", error);
        goto cleanup;
    }
    n_sim = datasets.n_sim;
    if (cli.has_max_sim && cli.max_sim < n_sim) n_sim = cli.max_sim;
    if (n_sim == 0) {
        fprintf(stderr, "No simulations were selected.\n");
        goto cleanup;
    }
    rows = (OutputRow *)calloc(n_sim, sizeof(*rows));
    if (rows == NULL) {
        fprintf(stderr, "Out of memory allocating result rows.\n");
        goto cleanup;
    }

    cm_run_options_init(&run_options);
    run_options.method = cli.method;
    run_options.n_mcmc = cli.n_mcmc;
    run_options.beta = cli.beta;
    run_options.k = cli.k;
    run_options.pstar = cli.pstar;
    run_options.profile = cli.profile;

    for (s = 0; s < n_sim; ++s) {
        CMDatasetView view;
        CMRng rng;
        CMStatus status;
        double sim_start;
        double prep_start;
        double method_start;
        double post_start;
        double standard_error;

        if (s == 0 ||
            (cli.print_every != 0 && (s + 1) % cli.print_every == 0) ||
            s + 1 == n_sim) {
            printf("Simulation %zu/%zu\n", s + 1, n_sim);
            fflush(stdout);
        }

        sim_start = cm_timer_now();
        prep_start = cm_timer_now();
        view.X = datasets.X + s * datasets.n_pop * datasets.n_cov;
        view.Y = datasets.Y + s * datasets.n_pop;
        view.g = datasets.g + s * datasets.n_pop;
        view.n_pop = datasets.n_pop;
        view.n_cov = datasets.n_cov;
        cm_rng_seed(&rng, datasets.seeds[s] + UINT64_C(1000000), UINT64_C(54));
        rows[s].time_sim_prep = cm_timer_now() - prep_start;

        method_start = cm_timer_now();
        status = cm_run_matching(&view, &run_options, &rng, &rows[s].result,
                                 error, sizeof(error));
        rows[s].time_method_call = cm_timer_now() - method_start;
        if (status != CM_STATUS_OK) {
            fprintf(stderr, "Simulation %zu failed (%s): %s\n",
                    s + 1, cm_status_string(status), error);
            goto cleanup;
        }

        post_start = cm_timer_now();
        if (isfinite(rows[s].result.mi_var) && rows[s].result.mi_var >= 0.0) {
            standard_error = sqrt(rows[s].result.mi_var);
            rows[s].has_ci = true;
            rows[s].ll = rows[s].result.est - 1.96 * standard_error;
            rows[s].ul = rows[s].result.est + 1.96 * standard_error;
            rows[s].covers_true =
                rows[s].ll <= cli.true_tau && cli.true_tau <= rows[s].ul;
        } else {
            rows[s].has_ci = false;
            rows[s].ll = NAN;
            rows[s].ul = NAN;
            rows[s].covers_true = 0;
        }
        rows[s].time_sim_post = cm_timer_now() - post_start;
        rows[s].time_sim_total = cm_timer_now() - sim_start;
        rows[s].share_method_in_sim = rows[s].time_sim_total > 0.0
            ? rows[s].time_method_call / rows[s].time_sim_total : 0.0;
    }

    if (!write_results_csv(output_path, rows, n_sim, &cli)) goto cleanup;
    printf("Saved results to %s\n", output_path);
    exit_code = EXIT_SUCCESS;

cleanup:
    free(rows);
    cm_dataset_free(&datasets);
    return exit_code;
}
