#include "cm_matching.h"

#include "cm_timer.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CMBaseData {
    const CMDatasetView *dataset;
    size_t *recipients;
    size_t *donors;
    size_t n_recipients;
    size_t n_donors;
    double *distance;
} CMBaseData;

typedef struct CMSeries {
    double *est;
    double *variance;
    double *cluster;
    double *paired;
    double *estdiff;
} CMSeries;

typedef struct CMSortEntry {
    double value;
    size_t index;
} CMSortEntry;

static void cm_set_error(char *buffer, size_t size, const char *format, ...)
{
    va_list args;

    if (buffer == NULL || size == 0) {
        return;
    }
    va_start(args, format);
    (void)vsnprintf(buffer, size, format, args);
    va_end(args);
    buffer[size - 1] = '\0';
}

static bool cm_checked_multiply(size_t a, size_t b, size_t *product)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }
    *product = a * b;
    return true;
}

static void *cm_allocate(size_t count,
                         size_t element_size,
                         bool clear,
                         CMStatus *status,
                         char *error,
                         size_t error_size)
{
    size_t bytes;
    void *memory;

    *status = CM_STATUS_OK;
    if (count == 0) {
        return NULL;
    }
    if (!cm_checked_multiply(count, element_size, &bytes)) {
        cm_set_error(error, error_size, "allocation size overflow");
        *status = CM_STATUS_OUT_OF_MEMORY;
        return NULL;
    }

    memory = clear ? calloc(count, element_size) : malloc(bytes);
    if (memory == NULL) {
        cm_set_error(error, error_size, "unable to allocate %zu bytes", bytes);
        *status = CM_STATUS_OUT_OF_MEMORY;
        return NULL;
    }
    return memory;
}

static int cm_compare_sort_entry(const void *left, const void *right)
{
    const CMSortEntry *a = (const CMSortEntry *)left;
    const CMSortEntry *b = (const CMSortEntry *)right;

    if (a->value < b->value) {
        return -1;
    }
    if (a->value > b->value) {
        return 1;
    }
    if (a->index < b->index) {
        return -1;
    }
    if (a->index > b->index) {
        return 1;
    }
    return 0;
}

static int cm_compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static int cm_compare_size_t(const void *left, const void *right)
{
    const size_t a = *(const size_t *)left;
    const size_t b = *(const size_t *)right;
    return (a > b) - (a < b);
}

static void cm_base_destroy(CMBaseData *base)
{
    if (base == NULL) {
        return;
    }
    free(base->recipients);
    free(base->donors);
    free(base->distance);
    memset(base, 0, sizeof(*base));
}

static CMStatus cm_base_initialize(const CMDatasetView *dataset,
                                   CMBaseData *base,
                                   char *error,
                                   size_t error_size)
{
    size_t i;
    size_t recipient_position = 0;
    size_t donor_position = 0;
    size_t distance_count;
    CMStatus status;

    memset(base, 0, sizeof(*base));
    base->dataset = dataset;

    for (i = 0; i < dataset->n_pop; ++i) {
        if (dataset->g[i] == 1) {
            ++base->n_recipients;
        } else {
            ++base->n_donors;
        }
    }

    base->recipients = cm_allocate(base->n_recipients,
                                   sizeof(*base->recipients), false, &status,
                                   error, error_size);
    if (status != CM_STATUS_OK) {
        goto fail;
    }
    base->donors = cm_allocate(base->n_donors, sizeof(*base->donors), false,
                               &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto fail;
    }

    for (i = 0; i < dataset->n_pop; ++i) {
        if (dataset->g[i] == 1) {
            base->recipients[recipient_position++] = i;
        } else {
            base->donors[donor_position++] = i;
        }
    }

    if (!cm_checked_multiply(base->n_recipients, base->n_donors,
                             &distance_count)) {
        cm_set_error(error, error_size, "distance matrix dimensions overflow");
        status = CM_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    base->distance = cm_allocate(distance_count, sizeof(*base->distance), false,
                                 &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto fail;
    }

    for (i = 0; i < base->n_recipients; ++i) {
        size_t j;
        const size_t recipient = base->recipients[i];
        for (j = 0; j < base->n_donors; ++j) {
            size_t covariate;
            const size_t donor = base->donors[j];
            double squared_distance = 0.0;
            for (covariate = 0; covariate < dataset->n_cov; ++covariate) {
                const double difference =
                    dataset->X[recipient * dataset->n_cov + covariate] -
                    dataset->X[donor * dataset->n_cov + covariate];
                squared_distance += difference * difference;
            }
            if (!isfinite(squared_distance)) {
                cm_set_error(error, error_size,
                             "non-finite Euclidean distance at recipient %zu, donor %zu",
                             i, j);
                status = CM_STATUS_NUMERICAL_ERROR;
                goto fail;
            }
            base->distance[i * base->n_donors + j] = sqrt(squared_distance);
        }
    }
    return CM_STATUS_OK;

fail:
    cm_base_destroy(base);
    return status;
}

static void cm_series_destroy(CMSeries *series)
{
    if (series == NULL) {
        return;
    }
    free(series->est);
    free(series->variance);
    free(series->cluster);
    free(series->paired);
    free(series->estdiff);
    memset(series, 0, sizeof(*series));
}

static CMStatus cm_series_initialize(CMSeries *series,
                                     size_t count,
                                     bool cluster,
                                     bool paired,
                                     bool estdiff,
                                     char *error,
                                     size_t error_size)
{
    CMStatus status;

    memset(series, 0, sizeof(*series));
#define CM_ALLOC_SERIES(member)                                                \
    do {                                                                        \
        series->member = cm_allocate(count, sizeof(*series->member), false,     \
                                     &status, error, error_size);                \
        if (status != CM_STATUS_OK) {                                            \
            goto fail;                                                          \
        }                                                                       \
    } while (0)

    CM_ALLOC_SERIES(est);
    CM_ALLOC_SERIES(variance);
    if (cluster) {
        CM_ALLOC_SERIES(cluster);
    }
    if (paired) {
        CM_ALLOC_SERIES(paired);
    }
    if (estdiff) {
        CM_ALLOC_SERIES(estdiff);
    }
#undef CM_ALLOC_SERIES
    return CM_STATUS_OK;

fail:
#undef CM_ALLOC_SERIES
    cm_series_destroy(series);
    return status;
}

static CMStatus cm_make_sorted_rows(const CMBaseData *base,
                                    size_t **rows_out,
                                    char *error,
                                    size_t error_size)
{
    size_t count;
    size_t *rows = NULL;
    CMSortEntry *entries = NULL;
    size_t r;
    CMStatus status;

    *rows_out = NULL;
    if (!cm_checked_multiply(base->n_recipients, base->n_donors, &count)) {
        cm_set_error(error, error_size, "sorted donor matrix dimensions overflow");
        return CM_STATUS_OUT_OF_MEMORY;
    }
    rows = cm_allocate(count, sizeof(*rows), false, &status,
                       error, error_size);
    if (status != CM_STATUS_OK) {
        return status;
    }
    entries = cm_allocate(base->n_donors, sizeof(*entries), false, &status,
                          error, error_size);
    if (status != CM_STATUS_OK) {
        free(rows);
        return status;
    }

    for (r = 0; r < base->n_recipients; ++r) {
        size_t d;
        for (d = 0; d < base->n_donors; ++d) {
            entries[d].value = base->distance[r * base->n_donors + d];
            entries[d].index = d;
        }
        qsort(entries, base->n_donors, sizeof(*entries), cm_compare_sort_entry);
        for (d = 0; d < base->n_donors; ++d) {
            rows[r * base->n_donors + d] = entries[d].index;
        }
    }
    free(entries);
    *rows_out = rows;
    return CM_STATUS_OK;
}

static CMStatus cm_make_alpha(const CMBaseData *base,
                              double beta,
                              double **alpha_out,
                              char *error,
                              size_t error_size)
{
    size_t count;
    double *alpha = NULL;
    size_t r;
    CMStatus status;

    *alpha_out = NULL;
    if (!cm_checked_multiply(base->n_recipients, base->n_donors, &count)) {
        cm_set_error(error, error_size, "probability matrix dimensions overflow");
        return CM_STATUS_OUT_OF_MEMORY;
    }
    alpha = cm_allocate(count, sizeof(*alpha), false, &status,
                        error, error_size);
    if (status != CM_STATUS_OK) {
        return status;
    }

    for (r = 0; r < base->n_recipients; ++r) {
        size_t d;
        double offset = base->distance[r * base->n_donors];
        double total = 0.0;
        for (d = 1; d < base->n_donors; ++d) {
            const double value = base->distance[r * base->n_donors + d];
            if (value < offset) {
                offset = value;
            }
        }
        for (d = 0; d < base->n_donors; ++d) {
            const double shifted =
                base->distance[r * base->n_donors + d] + offset;
            const double weight = 1.0 / pow(shifted, beta);
            if (!isfinite(weight) || weight < 0.0) {
                cm_set_error(error, error_size,
                             "non-finite donor weight for recipient %zu", r);
                free(alpha);
                return CM_STATUS_NUMERICAL_ERROR;
            }
            alpha[r * base->n_donors + d] = weight;
            total += weight;
        }
        if (!(total > 0.0) || !isfinite(total)) {
            cm_set_error(error, error_size,
                         "invalid donor weight total for recipient %zu", r);
            free(alpha);
            return CM_STATUS_NUMERICAL_ERROR;
        }
        for (d = 0; d < base->n_donors; ++d) {
            alpha[r * base->n_donors + d] /= total;
        }
    }

    *alpha_out = alpha;
    return CM_STATUS_OK;
}

static CMStatus cm_greedy_matches(const CMBaseData *base,
                                  size_t k,
                                  size_t **matches_out,
                                  size_t **unmatched_out,
                                  size_t *unmatched_count_out,
                                  char *error,
                                  size_t error_size)
{
    size_t *order = NULL;
    size_t *matches = NULL;
    size_t *unmatched = NULL;
    unsigned char *used = NULL;
    size_t match_count;
    size_t unmatched_count;
    size_t pass;
    CMStatus status;

    *matches_out = NULL;
    *unmatched_out = NULL;
    *unmatched_count_out = 0;
    if (!cm_checked_multiply(base->n_recipients, k, &match_count) ||
        match_count > base->n_donors) {
        cm_set_error(error, error_size, "infeasible constrained matching size");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    unmatched_count = base->n_donors - match_count;

    status = cm_make_sorted_rows(base, &order, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matches = cm_allocate(match_count, sizeof(*matches), false, &status,
                          error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    used = cm_allocate(base->n_donors, sizeof(*used), true, &status,
                       error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }

    for (pass = 0; pass < k; ++pass) {
        size_t r;
        for (r = 0; r < base->n_recipients; ++r) {
            size_t position;
            bool found = false;
            for (position = 0; position < base->n_donors; ++position) {
                const size_t donor = order[r * base->n_donors + position];
                if (!used[donor]) {
                    matches[r * k + pass] = donor;
                    used[donor] = 1;
                    found = true;
                    break;
                }
            }
            if (!found) {
                cm_set_error(error, error_size,
                             "greedy initializer ran out of donors");
                status = CM_STATUS_INTERNAL_ERROR;
                goto cleanup;
            }
        }
    }

    /* np.where(C_acc[row] == 1) returns each row in donor-index order. */
    if (k > 1) {
        size_t r;
        for (r = 0; r < base->n_recipients; ++r) {
            qsort(matches + r * k, k, sizeof(*matches), cm_compare_size_t);
        }
    }

    unmatched = cm_allocate(unmatched_count, sizeof(*unmatched), false,
                            &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    if (unmatched_count > 0) {
        size_t donor;
        size_t position = 0;
        for (donor = 0; donor < base->n_donors; ++donor) {
            if (!used[donor]) {
                unmatched[position++] = donor;
            }
        }
    }

    *matches_out = matches;
    *unmatched_out = unmatched;
    *unmatched_count_out = unmatched_count;
    matches = NULL;
    unmatched = NULL;
    status = CM_STATUS_OK;

cleanup:
    free(order);
    free(matches);
    free(unmatched);
    free(used);
    return status;
}

static double cm_mean(const double *values, size_t count)
{
    size_t i;
    double total = 0.0;
    for (i = 0; i < count; ++i) {
        total += values[i];
    }
    return total / (double)count;
}

static void cm_ols_binary(const double *treated,
                          size_t n_treated,
                          const double *controls,
                          size_t n_controls,
                          double *slope,
                          double *slope_variance)
{
    const double treated_mean = cm_mean(treated, n_treated);
    const double control_mean = cm_mean(controls, n_controls);
    double residual_ss = 0.0;
    size_t i;

    for (i = 0; i < n_treated; ++i) {
        const double residual = treated[i] - treated_mean;
        residual_ss += residual * residual;
    }
    for (i = 0; i < n_controls; ++i) {
        const double residual = controls[i] - control_mean;
        residual_ss += residual * residual;
    }

    *slope = treated_mean - control_mean;
    *slope_variance =
        residual_ss / (double)(n_treated + n_controls - 2) *
        (1.0 / (double)n_treated + 1.0 / (double)n_controls);
}

static void cm_wls_1tok(const double *treated,
                        size_t n_recipients,
                        const double *controls,
                        size_t k,
                        double *slope,
                        double *slope_variance)
{
    const size_t n_controls = n_recipients * k;
    const double treated_mean = cm_mean(treated, n_recipients);
    const double control_mean = cm_mean(controls, n_controls);
    double residual_ss = 0.0;
    size_t i;

    for (i = 0; i < n_recipients; ++i) {
        const double residual = treated[i] - treated_mean;
        residual_ss += residual * residual;
    }
    for (i = 0; i < n_controls; ++i) {
        const double residual = controls[i] - control_mean;
        residual_ss += residual * residual / (double)k;
    }

    *slope = treated_mean - control_mean;
    *slope_variance =
        residual_ss / (double)(n_recipients + n_controls - 2) *
        (2.0 / (double)n_recipients);
}

static double cm_paired_variance(const double *treated,
                                 const double *matches,
                                 size_t n_recipients,
                                 size_t k)
{
    size_t r;
    double difference_mean = 0.0;
    double sum_squares = 0.0;

    for (r = 0; r < n_recipients; ++r) {
        size_t slot;
        double match_mean = 0.0;
        for (slot = 0; slot < k; ++slot) {
            match_mean += matches[r * k + slot];
        }
        difference_mean += treated[r] - match_mean / (double)k;
    }
    difference_mean /= (double)n_recipients;

    for (r = 0; r < n_recipients; ++r) {
        size_t slot;
        double match_mean = 0.0;
        double centered;
        for (slot = 0; slot < k; ++slot) {
            match_mean += matches[r * k + slot];
        }
        centered = treated[r] - match_mean / (double)k - difference_mean;
        sum_squares += centered * centered;
    }
    return sum_squares /
           (double)(n_recipients - 1) /
           (double)n_recipients;
}

static double cm_cluster_se_1tok(const double *treated,
                                 const double *matches,
                                 size_t n_recipients,
                                 size_t k)
{
    const size_t n_controls = n_recipients * k;
    const double treated_mean = cm_mean(treated, n_recipients);
    const double control_mean = cm_mean(matches, n_controls);
    double variance = 0.0;
    size_t r;

    for (r = 0; r < n_recipients; ++r) {
        size_t slot;
        double control_residual_sum = 0.0;
        double influence;
        for (slot = 0; slot < k; ++slot) {
            control_residual_sum += matches[r * k + slot] - control_mean;
        }
        influence =
            (treated[r] - treated_mean) / (double)n_recipients -
            control_residual_sum / (double)n_controls;
        variance += influence * influence;
    }
    return sqrt(fmax(variance, 0.0));
}

static void cm_mi_total_variance(const double *estimates,
                                 const double *within,
                                 size_t count,
                                 double *estimate_mean,
                                 double *within_mean,
                                 double *between,
                                 double *total)
{
    size_t i;
    double squared_deviation = 0.0;

    *estimate_mean = cm_mean(estimates, count);
    *within_mean = cm_mean(within, count);
    for (i = 0; i < count; ++i) {
        const double deviation = estimates[i] - *estimate_mean;
        squared_deviation += deviation * deviation;
    }
    *between = count > 1
        ? squared_deviation / (double)(count - 1)
        : NAN;
    *total = *within_mean + (1.0 + 1.0 / (double)count) * *between;
}

static double cm_median_in_place(double *values, size_t count)
{
    qsort(values, count, sizeof(*values), cm_compare_double);
    if ((count & 1u) != 0u) {
        return values[count / 2];
    }
    return (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

static double cm_safe_ratio(double numerator, double denominator)
{
    return denominator != 0.0 ? numerator / denominator : NAN;
}

static void cm_fill_timing(CMMethodResult *result,
                           const CMRunOptions *options,
                           const CMMethodTiming *measured,
                           uint64_t n_sampling_draws)
{
    CMMethodTiming timing;

    if (!options->profile) {
        memset(&result->timing, 0, sizeof(result->timing));
        return;
    }

    timing = *measured;
    timing.mcmc_total = timing.mcmc_sampling + timing.mcmc_estimation;
    timing.method_total = timing.setup + timing.allocation +
                          timing.mcmc_sampling + timing.mcmc_estimation +
                          timing.summary;
    timing.n_sampling_draws = n_sampling_draws;
    timing.time_per_sampling_draw =
        n_sampling_draws != 0
            ? timing.mcmc_sampling / (double)n_sampling_draws
            : NAN;
    timing.share_sampling_in_mcmc =
        cm_safe_ratio(timing.mcmc_sampling, timing.mcmc_total);
    timing.share_sampling_in_method =
        cm_safe_ratio(timing.mcmc_sampling, timing.method_total);
    timing.share_estimation_in_method =
        cm_safe_ratio(timing.mcmc_estimation, timing.method_total);
    timing.share_mcmc_in_method =
        cm_safe_ratio(timing.mcmc_total, timing.method_total);
    result->timing = timing;
    result->flags |= CM_RESULT_HAS_PROFILE;
}

static void cm_fill_common_summary(CMMethodResult *result,
                                   CMSeries *series,
                                   size_t count)
{
    cm_mi_total_variance(series->est, series->variance, count,
                         &result->est, &result->wi_var,
                         &result->bw_var, &result->mi_var);
}

static void cm_fill_paired_summary(CMMethodResult *result,
                                   const CMSeries *series,
                                   size_t count)
{
    double ignored_estimate;
    double ignored_between;
    cm_mi_total_variance(series->est, series->paired, count,
                         &ignored_estimate, &result->wi_var_paired,
                         &ignored_between, &result->mi_var_paired);
    result->flags |= CM_RESULT_HAS_PAIRED_VARIANCE;
}

static void cm_fill_cluster_summary(CMMethodResult *result,
                                    const CMSeries *series,
                                    size_t count)
{
    double ignored_estimate;
    double ignored_between;
    cm_mi_total_variance(series->est, series->cluster, count,
                         &ignored_estimate, &result->wi_var_clus,
                         &ignored_between, &result->mi_var_clus);
    result->flags |= CM_RESULT_HAS_CLUSTER_VARIANCE;
}

static void cm_copy_recipient_outcomes(const CMBaseData *base, double *outcomes)
{
    size_t r;
    for (r = 0; r < base->n_recipients; ++r) {
        outcomes[r] = base->dataset->Y[base->recipients[r]];
    }
}

static double cm_log_uniform(CMRng *rng)
{
    return log(cm_rng_uniform(rng));
}

static CMStatus cm_run_unconstrained_1to1(const CMDatasetView *dataset,
                                          const CMRunOptions *options,
                                          CMRng *rng,
                                          CMMethodResult *result,
                                          char *error,
                                          size_t error_size)
{
    CMBaseData base;
    CMSeries series;
    CMMethodTiming timing;
    double *cdf = NULL;
    double *recipient_outcomes = NULL;
    double *matched_outcomes = NULL;
    double started;
    size_t iteration;
    CMStatus status;

    memset(&base, 0, sizeof(base));
    memset(&series, 0, sizeof(series));
    memset(&timing, 0, sizeof(timing));

    started = cm_timer_now();
    status = cm_base_initialize(dataset, &base, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_make_alpha(&base, options->beta, &cdf, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    {
        size_t r;
        for (r = 0; r < base.n_recipients; ++r) {
            size_t d;
            double cumulative = 0.0;
            for (d = 0; d < base.n_donors; ++d) {
                cumulative += cdf[r * base.n_donors + d];
                cdf[r * base.n_donors + d] = cumulative;
            }
            cdf[r * base.n_donors + base.n_donors - 1] = 1.0;
        }
    }
    timing.setup += cm_timer_now() - started;

    started = cm_timer_now();
    status = cm_series_initialize(&series, options->n_mcmc, false, true, true,
                                  error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    recipient_outcomes = cm_allocate(base.n_recipients,
                                     sizeof(*recipient_outcomes), false,
                                     &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matched_outcomes = cm_allocate(base.n_recipients,
                                   sizeof(*matched_outcomes), false,
                                   &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cm_copy_recipient_outcomes(&base, recipient_outcomes);
    timing.allocation += cm_timer_now() - started;

    for (iteration = 0; iteration < options->n_mcmc; ++iteration) {
        size_t r;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            const double uniform = cm_rng_uniform(rng);
            size_t donor = 0;
            while (donor + 1 < base.n_donors &&
                   uniform > cdf[r * base.n_donors + donor]) {
                ++donor;
            }
            matched_outcomes[r] =
                dataset->Y[base.donors[donor]];
        }
        timing.mcmc_sampling += cm_timer_now() - started;

        started = cm_timer_now();
        cm_ols_binary(recipient_outcomes, base.n_recipients,
                      matched_outcomes, base.n_recipients,
                      &series.est[iteration], &series.variance[iteration]);
        series.estdiff[iteration] =
            cm_mean(recipient_outcomes, base.n_recipients) -
            cm_mean(matched_outcomes, base.n_recipients);
        series.paired[iteration] =
            cm_paired_variance(recipient_outcomes, matched_outcomes,
                               base.n_recipients, 1);
        timing.mcmc_estimation += cm_timer_now() - started;
    }

    started = cm_timer_now();
    cm_fill_common_summary(result, &series, options->n_mcmc);
    cm_fill_paired_summary(result, &series, options->n_mcmc);
    timing.summary += cm_timer_now() - started;

    result->est_med = cm_median_in_place(series.est, options->n_mcmc);
    result->estdiff = cm_mean(series.estdiff, options->n_mcmc);
    result->flags |= CM_RESULT_HAS_ESTDIFF;
    cm_fill_timing(result, options, &timing,
                   (uint64_t)(options->n_mcmc * base.n_recipients));
    status = CM_STATUS_OK;

cleanup:
    free(cdf);
    free(recipient_outcomes);
    free(matched_outcomes);
    cm_series_destroy(&series);
    cm_base_destroy(&base);
    return status;
}

static CMStatus cm_run_unconstrained_1tok(const CMDatasetView *dataset,
                                          const CMRunOptions *options,
                                          CMRng *rng,
                                          CMMethodResult *result,
                                          char *error,
                                          size_t error_size)
{
    CMBaseData base;
    CMSeries series;
    CMMethodTiming timing;
    double *alpha = NULL;
    size_t *donor_order = NULL;
    double *recipient_outcomes = NULL;
    double *matched_outcomes = NULL;
    double started;
    size_t iteration;
    uint64_t acceptance_count = 0;
    size_t match_count = 0;
    CMStatus status;

    memset(&base, 0, sizeof(base));
    memset(&series, 0, sizeof(series));
    memset(&timing, 0, sizeof(timing));

    started = cm_timer_now();
    status = cm_base_initialize(dataset, &base, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_make_alpha(&base, options->beta, &alpha, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_make_sorted_rows(&base, &donor_order, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    timing.setup += cm_timer_now() - started;

    started = cm_timer_now();
    status = cm_series_initialize(&series, options->n_mcmc, false, true, true,
                                  error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    if (!cm_checked_multiply(base.n_recipients, options->k, &match_count)) {
        cm_set_error(error, error_size, "matched outcome dimensions overflow");
        status = CM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    recipient_outcomes = cm_allocate(base.n_recipients,
                                     sizeof(*recipient_outcomes), false,
                                     &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matched_outcomes = cm_allocate(match_count, sizeof(*matched_outcomes), false,
                                   &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cm_copy_recipient_outcomes(&base, recipient_outcomes);
    timing.allocation += cm_timer_now() - started;

    for (iteration = 0; iteration < options->n_mcmc; ++iteration) {
        bool accepted_any = false;
        size_t r;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            const size_t remove_position = cm_rng_bounded(rng, options->k);
            const size_t add_position = options->k +
                cm_rng_bounded(rng, base.n_donors - options->k);
            const size_t x1 = donor_order[r * base.n_donors + remove_position];
            const size_t x0 = donor_order[r * base.n_donors + add_position];
            const double probability_x1 = alpha[r * base.n_donors + x1];
            const double probability_x0 = alpha[r * base.n_donors + x0];
            const double weight0 = probability_x0 / (1.0 - probability_x0);
            const double weight1 = probability_x1 / (1.0 - probability_x1);
            const double log_ratio = log(weight0 / weight1);
            const double log_acceptance = fmin(0.0, log_ratio);
            size_t slot;

            if (cm_log_uniform(rng) < log_acceptance) {
                donor_order[r * base.n_donors + remove_position] = x0;
                donor_order[r * base.n_donors + add_position] = x1;
                accepted_any = true;
            }
            for (slot = 0; slot < options->k; ++slot) {
                const size_t donor =
                    donor_order[r * base.n_donors + slot];
                matched_outcomes[r * options->k + slot] =
                    dataset->Y[base.donors[donor]];
            }
        }
        timing.mcmc_sampling += cm_timer_now() - started;
        if (accepted_any) {
            ++acceptance_count;
        }

        started = cm_timer_now();
        cm_ols_binary(recipient_outcomes, base.n_recipients,
                      matched_outcomes, match_count,
                      &series.est[iteration], &series.variance[iteration]);
        series.estdiff[iteration] =
            cm_mean(recipient_outcomes, base.n_recipients) -
            cm_mean(matched_outcomes, match_count);
        series.paired[iteration] =
            cm_paired_variance(recipient_outcomes, matched_outcomes,
                               base.n_recipients, options->k);
        timing.mcmc_estimation += cm_timer_now() - started;
    }

    started = cm_timer_now();
    cm_fill_common_summary(result, &series, options->n_mcmc);
    cm_fill_paired_summary(result, &series, options->n_mcmc);
    timing.summary += cm_timer_now() - started;

    result->est_med = cm_median_in_place(series.est, options->n_mcmc);
    result->estdiff = cm_mean(series.estdiff, options->n_mcmc);
    result->k = options->k;
    result->acceptance_count = acceptance_count;
    result->flags |= CM_RESULT_HAS_K | CM_RESULT_HAS_ESTDIFF |
                     CM_RESULT_HAS_ACCEPTANCE_COUNT;
    cm_fill_timing(result, options, &timing,
                   (uint64_t)(options->n_mcmc * base.n_recipients));
    status = CM_STATUS_OK;

cleanup:
    free(alpha);
    free(donor_order);
    free(recipient_outcomes);
    free(matched_outcomes);
    cm_series_destroy(&series);
    cm_base_destroy(&base);
    return status;
}

static size_t cm_dp_index(size_t recipient,
                          size_t donor,
                          size_t remaining,
                          size_t n_donors,
                          size_t k)
{
    return (recipient * (n_donors + 1) + donor) * (k + 1) + remaining;
}

static CMStatus cm_run_unconstrained_1tok_mc(const CMDatasetView *dataset,
                                             const CMRunOptions *options,
                                             CMRng *rng,
                                             CMMethodResult *result,
                                             char *error,
                                             size_t error_size)
{
    CMBaseData base;
    CMSeries series;
    CMMethodTiming timing;
    double *row_weights = NULL;
    double *dp = NULL;
    double *recipient_outcomes = NULL;
    double *matched_outcomes = NULL;
    size_t match_count = 0;
    size_t dp_rows = 0;
    size_t dp_count = 0;
    double started;
    size_t iteration;
    CMStatus status;

    memset(&base, 0, sizeof(base));
    memset(&series, 0, sizeof(series));
    memset(&timing, 0, sizeof(timing));

    started = cm_timer_now();
    status = cm_base_initialize(dataset, &base, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_make_alpha(&base, options->beta, &row_weights,
                           error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    {
        size_t r;
        for (r = 0; r < base.n_recipients; ++r) {
            size_t d;
            for (d = 0; d < base.n_donors; ++d) {
                double *weight = &row_weights[r * base.n_donors + d];
                *weight /= 1.0 - *weight;
            }
        }
    }
    if (!cm_checked_multiply(base.n_recipients, base.n_donors + 1,
                             &dp_rows) ||
        !cm_checked_multiply(dp_rows, options->k + 1, &dp_count)) {
        cm_set_error(error, error_size, "dynamic-programming table dimensions overflow");
        status = CM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    dp = cm_allocate(dp_count, sizeof(*dp), true, &status,
                     error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    {
        size_t r;
        for (r = 0; r < base.n_recipients; ++r) {
            size_t donor;
            for (donor = 0; donor <= base.n_donors; ++donor) {
                dp[cm_dp_index(r, donor, 0, base.n_donors, options->k)] = 1.0;
            }
            for (donor = base.n_donors; donor-- > 0;) {
                size_t remaining;
                for (remaining = 1; remaining <= options->k; ++remaining) {
                    const double value =
                        dp[cm_dp_index(r, donor + 1, remaining,
                                      base.n_donors, options->k)] +
                        row_weights[r * base.n_donors + donor] *
                        dp[cm_dp_index(r, donor + 1, remaining - 1,
                                      base.n_donors, options->k)];
                    if (!isfinite(value)) {
                        cm_set_error(error, error_size,
                                     "dynamic-programming weights overflow for recipient %zu", r);
                        status = CM_STATUS_NUMERICAL_ERROR;
                        goto cleanup;
                    }
                    dp[cm_dp_index(r, donor, remaining,
                                   base.n_donors, options->k)] = value;
                }
            }
        }
    }
    timing.setup += cm_timer_now() - started;

    started = cm_timer_now();
    status = cm_series_initialize(&series, options->n_mcmc, false, true, true,
                                  error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    if (!cm_checked_multiply(base.n_recipients, options->k, &match_count)) {
        cm_set_error(error, error_size, "matched outcome dimensions overflow");
        status = CM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    recipient_outcomes = cm_allocate(base.n_recipients,
                                     sizeof(*recipient_outcomes), false,
                                     &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matched_outcomes = cm_allocate(match_count, sizeof(*matched_outcomes), false,
                                   &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cm_copy_recipient_outcomes(&base, recipient_outcomes);
    timing.allocation += cm_timer_now() - started;

    for (iteration = 0; iteration < options->n_mcmc; ++iteration) {
        size_t r;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            size_t remaining = options->k;
            size_t selected = 0;
            size_t donor;
            for (donor = 0; donor < base.n_donors && remaining != 0; ++donor) {
                const double denominator =
                    dp[cm_dp_index(r, donor, remaining,
                                   base.n_donors, options->k)];
                const double numerator =
                    row_weights[r * base.n_donors + donor] *
                    dp[cm_dp_index(r, donor + 1, remaining - 1,
                                   base.n_donors, options->k)];
                const double include_probability = numerator / denominator;
                if (!isfinite(include_probability)) {
                    cm_set_error(error, error_size,
                                 "invalid conditional probability for recipient %zu", r);
                    status = CM_STATUS_NUMERICAL_ERROR;
                    goto cleanup;
                }
                if (cm_rng_uniform(rng) < include_probability) {
                    matched_outcomes[r * options->k + selected] =
                        dataset->Y[base.donors[donor]];
                    ++selected;
                    --remaining;
                }
            }
            if (remaining != 0) {
                cm_set_error(error, error_size,
                             "Monte Carlo row sampler failed to select k donors");
                status = CM_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
        }
        timing.mcmc_sampling += cm_timer_now() - started;

        started = cm_timer_now();
        cm_ols_binary(recipient_outcomes, base.n_recipients,
                      matched_outcomes, match_count,
                      &series.est[iteration], &series.variance[iteration]);
        series.estdiff[iteration] =
            cm_mean(recipient_outcomes, base.n_recipients) -
            cm_mean(matched_outcomes, match_count);
        series.paired[iteration] =
            cm_paired_variance(recipient_outcomes, matched_outcomes,
                               base.n_recipients, options->k);
        timing.mcmc_estimation += cm_timer_now() - started;
    }

    started = cm_timer_now();
    cm_fill_common_summary(result, &series, options->n_mcmc);
    cm_fill_paired_summary(result, &series, options->n_mcmc);
    timing.summary += cm_timer_now() - started;

    result->est_med = cm_median_in_place(series.est, options->n_mcmc);
    result->estdiff = cm_mean(series.estdiff, options->n_mcmc);
    result->k = options->k;
    result->flags |= CM_RESULT_HAS_K | CM_RESULT_HAS_ESTDIFF;
    cm_fill_timing(result, options, &timing,
                   (uint64_t)(options->n_mcmc * base.n_recipients));
    status = CM_STATUS_OK;

cleanup:
    free(row_weights);
    free(dp);
    free(recipient_outcomes);
    free(matched_outcomes);
    cm_series_destroy(&series);
    cm_base_destroy(&base);
    return status;
}

static CMStatus cm_run_constrained_1to1(const CMDatasetView *dataset,
                                        const CMRunOptions *options,
                                        CMRng *rng,
                                        CMMethodResult *result,
                                        char *error,
                                        size_t error_size)
{
    CMBaseData base;
    CMSeries series;
    CMMethodTiming timing;
    size_t *matches = NULL;
    size_t *unmatched = NULL;
    size_t unmatched_count = 0;
    double *recipient_outcomes = NULL;
    double *matched_outcomes = NULL;
    double started;
    size_t iteration;
    uint64_t acceptance_count = 0;
    CMStatus status;

    memset(&base, 0, sizeof(base));
    memset(&series, 0, sizeof(series));
    memset(&timing, 0, sizeof(timing));

    started = cm_timer_now();
    status = cm_base_initialize(dataset, &base, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_greedy_matches(&base, 1, &matches, &unmatched,
                               &unmatched_count, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    timing.setup += cm_timer_now() - started;

    started = cm_timer_now();
    status = cm_series_initialize(&series, options->n_mcmc, true, true, true,
                                  error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    recipient_outcomes = cm_allocate(base.n_recipients,
                                     sizeof(*recipient_outcomes), false,
                                     &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matched_outcomes = cm_allocate(base.n_recipients,
                                   sizeof(*matched_outcomes), false,
                                   &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cm_copy_recipient_outcomes(&base, recipient_outcomes);
    timing.allocation += cm_timer_now() - started;

    for (iteration = 0; iteration < options->n_mcmc; ++iteration) {
        bool swap_rows;
        bool accepted = false;
        size_t row1 = 0;
        size_t row2 = 0;
        size_t old1 = 0;
        size_t old2 = 0;
        size_t unmatched_position = 0;
        size_t new_donor = 0;
        size_t recipient = 0;
        size_t old_donor = 0;
        double log_likelihood_new;
        double log_likelihood_old;
        double log_acceptance;
        size_t r;

        started = cm_timer_now();
        swap_rows = cm_rng_uniform(rng) < options->pstar;
        if (swap_rows) {
            row1 = cm_rng_bounded(rng, base.n_recipients);
            row2 = cm_rng_bounded(rng, base.n_recipients - 1);
            if (row2 >= row1) {
                ++row2;
            }
            old1 = matches[row1];
            old2 = matches[row2];
            log_likelihood_new = -options->beta *
                (base.distance[row1 * base.n_donors + old2] +
                 base.distance[row2 * base.n_donors + old1]);
            log_likelihood_old = -options->beta *
                (base.distance[row1 * base.n_donors + old1] +
                 base.distance[row2 * base.n_donors + old2]);
        } else if (unmatched_count == 0) {
            log_likelihood_new = -INFINITY;
            log_likelihood_old = 0.0;
        } else {
            unmatched_position = cm_rng_bounded(rng, unmatched_count);
            new_donor = unmatched[unmatched_position];
            recipient = cm_rng_bounded(rng, base.n_recipients);
            old_donor = matches[recipient];
            log_likelihood_new = -options->beta *
                base.distance[recipient * base.n_donors + new_donor];
            log_likelihood_old = -options->beta *
                base.distance[recipient * base.n_donors + old_donor];
        }
        log_acceptance = fmin(0.0, log_likelihood_new - log_likelihood_old);
        if (cm_log_uniform(rng) < log_acceptance) {
            if (swap_rows) {
                matches[row1] = old2;
                matches[row2] = old1;
                accepted = true;
            } else if (unmatched_count > 0) {
                matches[recipient] = new_donor;
                unmatched[unmatched_position] = old_donor;
                accepted = true;
            }
        }
        if (accepted) {
            ++acceptance_count;
        }
        timing.mcmc_sampling += cm_timer_now() - started;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            matched_outcomes[r] = dataset->Y[base.donors[matches[r]]];
        }
        cm_ols_binary(recipient_outcomes, base.n_recipients,
                      matched_outcomes, base.n_recipients,
                      &series.est[iteration], &series.variance[iteration]);
        series.estdiff[iteration] =
            cm_mean(recipient_outcomes, base.n_recipients) -
            cm_mean(matched_outcomes, base.n_recipients);
        /* Python stores this standard error in its within-variance vector. */
        series.cluster[iteration] =
            cm_cluster_se_1tok(recipient_outcomes, matched_outcomes,
                               base.n_recipients, 1);
        series.paired[iteration] =
            cm_paired_variance(recipient_outcomes, matched_outcomes,
                               base.n_recipients, 1);
        timing.mcmc_estimation += cm_timer_now() - started;
    }

    started = cm_timer_now();
    cm_fill_common_summary(result, &series, options->n_mcmc);
    cm_fill_cluster_summary(result, &series, options->n_mcmc);
    cm_fill_paired_summary(result, &series, options->n_mcmc);
    timing.summary += cm_timer_now() - started;

    result->est_med = cm_median_in_place(series.est, options->n_mcmc);
    result->estdiff = cm_mean(series.estdiff, options->n_mcmc);
    result->acceptance_count = acceptance_count;
    result->flags |= CM_RESULT_HAS_ESTDIFF | CM_RESULT_HAS_ACCEPTANCE_COUNT;
    cm_fill_timing(result, options, &timing, (uint64_t)options->n_mcmc);
    status = CM_STATUS_OK;

cleanup:
    free(matches);
    free(unmatched);
    free(recipient_outcomes);
    free(matched_outcomes);
    cm_series_destroy(&series);
    cm_base_destroy(&base);
    return status;
}

static CMStatus cm_run_constrained_1to1_gibbs(const CMDatasetView *dataset,
                                              const CMRunOptions *options,
                                              CMRng *rng,
                                              CMMethodResult *result,
                                              char *error,
                                              size_t error_size)
{
    CMBaseData base;
    CMSeries series;
    CMMethodTiming timing;
    size_t *matches = NULL;
    size_t *unmatched = NULL;
    size_t unmatched_count = 0;
    double *base_weights = NULL;
    double *weights = NULL;
    double *cdf = NULL;
    double *recipient_outcomes = NULL;
    double *matched_outcomes = NULL;
    size_t distance_count = 0;
    double started;
    size_t iteration;
    uint64_t change_count = 0;
    uint64_t changed_iteration_count = 0;
    CMStatus status;

    memset(&base, 0, sizeof(base));
    memset(&series, 0, sizeof(series));
    memset(&timing, 0, sizeof(timing));

    started = cm_timer_now();
    status = cm_base_initialize(dataset, &base, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_greedy_matches(&base, 1, &matches, &unmatched,
                               &unmatched_count, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    if (!cm_checked_multiply(base.n_recipients, base.n_donors,
                             &distance_count)) {
        cm_set_error(error, error_size, "base weight dimensions overflow");
        status = CM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    base_weights = cm_allocate(distance_count, sizeof(*base_weights), false,
                               &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    {
        size_t r;
        for (r = 0; r < base.n_recipients; ++r) {
            size_t d;
            double minimum = base.distance[r * base.n_donors];
            for (d = 1; d < base.n_donors; ++d) {
                const double value = base.distance[r * base.n_donors + d];
                if (value < minimum) {
                    minimum = value;
                }
            }
            for (d = 0; d < base.n_donors; ++d) {
                base_weights[r * base.n_donors + d] =
                    exp(-options->beta *
                        (base.distance[r * base.n_donors + d] - minimum));
            }
        }
    }
    timing.setup += cm_timer_now() - started;

    started = cm_timer_now();
    status = cm_series_initialize(&series, options->n_mcmc, true, true, true,
                                  error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    recipient_outcomes = cm_allocate(base.n_recipients,
                                     sizeof(*recipient_outcomes), false,
                                     &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matched_outcomes = cm_allocate(base.n_recipients,
                                   sizeof(*matched_outcomes), false,
                                   &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    weights = cm_allocate(unmatched_count + 1, sizeof(*weights), false,
                          &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cdf = cm_allocate(unmatched_count + 1, sizeof(*cdf), false,
                      &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cm_copy_recipient_outcomes(&base, recipient_outcomes);
    timing.allocation += cm_timer_now() - started;

    for (iteration = 0; iteration < options->n_mcmc; ++iteration) {
        bool changed_any = false;
        size_t r;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            const size_t current = matches[r];
            double cumulative;
            double target;
            size_t position;
            size_t slot = 0;

            weights[0] = base_weights[r * base.n_donors + current];
            for (position = 0; position < unmatched_count; ++position) {
                weights[position + 1] =
                    base_weights[r * base.n_donors + unmatched[position]];
            }
            cumulative = 0.0;
            for (position = 0; position <= unmatched_count; ++position) {
                cumulative += weights[position];
                cdf[position] = cumulative;
            }
            if (!(cumulative > 0.0) || !isfinite(cumulative)) {
                cm_set_error(error, error_size,
                             "invalid Gibbs conditional weight total for recipient %zu", r);
                status = CM_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            target = cm_rng_uniform(rng) * cumulative;
            while (slot <= unmatched_count && cdf[slot] <= target) {
                ++slot;
            }
            if (slot > unmatched_count) {
                cm_set_error(error, error_size,
                             "Gibbs conditional search exceeded its support");
                status = CM_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            if (slot > 0) {
                const size_t unmatched_position = slot - 1;
                const size_t new_donor = unmatched[unmatched_position];
                matches[r] = new_donor;
                unmatched[unmatched_position] = current;
                changed_any = true;
                ++change_count;
            }
        }
        if (changed_any) {
            ++changed_iteration_count;
        }
        timing.mcmc_sampling += cm_timer_now() - started;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            matched_outcomes[r] = dataset->Y[base.donors[matches[r]]];
        }
        cm_ols_binary(recipient_outcomes, base.n_recipients,
                      matched_outcomes, base.n_recipients,
                      &series.est[iteration], &series.variance[iteration]);
        series.estdiff[iteration] =
            cm_mean(recipient_outcomes, base.n_recipients) -
            cm_mean(matched_outcomes, base.n_recipients);
        series.cluster[iteration] =
            cm_cluster_se_1tok(recipient_outcomes, matched_outcomes,
                               base.n_recipients, 1);
        series.paired[iteration] =
            cm_paired_variance(recipient_outcomes, matched_outcomes,
                               base.n_recipients, 1);
        timing.mcmc_estimation += cm_timer_now() - started;
    }

    started = cm_timer_now();
    cm_fill_common_summary(result, &series, options->n_mcmc);
    cm_fill_cluster_summary(result, &series, options->n_mcmc);
    cm_fill_paired_summary(result, &series, options->n_mcmc);
    timing.summary += cm_timer_now() - started;

    result->est_med = cm_median_in_place(series.est, options->n_mcmc);
    result->estdiff = cm_mean(series.estdiff, options->n_mcmc);
    result->change_count = change_count;
    result->changed_iteration_count = changed_iteration_count;
    result->flags |= CM_RESULT_HAS_ESTDIFF | CM_RESULT_HAS_CHANGE_COUNT |
                     CM_RESULT_HAS_CHANGED_ITERATION_COUNT;
    cm_fill_timing(result, options, &timing,
                   (uint64_t)(options->n_mcmc * base.n_recipients));
    status = CM_STATUS_OK;

cleanup:
    free(matches);
    free(unmatched);
    free(base_weights);
    free(weights);
    free(cdf);
    free(recipient_outcomes);
    free(matched_outcomes);
    cm_series_destroy(&series);
    cm_base_destroy(&base);
    return status;
}

static CMStatus cm_run_constrained_1tok(const CMDatasetView *dataset,
                                        const CMRunOptions *options,
                                        CMRng *rng,
                                        CMMethodResult *result,
                                        char *error,
                                        size_t error_size)
{
    CMBaseData base;
    CMSeries series;
    CMMethodTiming timing;
    size_t *matches = NULL;
    size_t *unmatched = NULL;
    size_t unmatched_count = 0;
    double *recipient_outcomes = NULL;
    double *matched_outcomes = NULL;
    size_t match_count = 0;
    double started;
    size_t iteration;
    uint64_t acceptance_count = 0;
    CMStatus status;

    memset(&base, 0, sizeof(base));
    memset(&series, 0, sizeof(series));
    memset(&timing, 0, sizeof(timing));

    started = cm_timer_now();
    status = cm_base_initialize(dataset, &base, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    status = cm_greedy_matches(&base, options->k, &matches, &unmatched,
                               &unmatched_count, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    if (!cm_checked_multiply(base.n_recipients, options->k, &match_count)) {
        cm_set_error(error, error_size, "matched outcome dimensions overflow");
        status = CM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    timing.setup += cm_timer_now() - started;

    started = cm_timer_now();
    status = cm_series_initialize(&series, options->n_mcmc, true, false, false,
                                  error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    recipient_outcomes = cm_allocate(base.n_recipients,
                                     sizeof(*recipient_outcomes), false,
                                     &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    matched_outcomes = cm_allocate(match_count, sizeof(*matched_outcomes), false,
                                   &status, error, error_size);
    if (status != CM_STATUS_OK) {
        goto cleanup;
    }
    cm_copy_recipient_outcomes(&base, recipient_outcomes);
    timing.allocation += cm_timer_now() - started;

    for (iteration = 0; iteration < options->n_mcmc; ++iteration) {
        bool swap_rows;
        bool accepted = false;
        size_t row1 = 0;
        size_t row2 = 0;
        size_t slot1 = 0;
        size_t slot2 = 0;
        size_t old1 = 0;
        size_t old2 = 0;
        size_t unmatched_position = 0;
        size_t new_donor = 0;
        size_t recipient = 0;
        size_t slot = 0;
        size_t old_donor = 0;
        double log_likelihood_new;
        double log_likelihood_old;
        double log_acceptance;
        size_t r;

        started = cm_timer_now();
        swap_rows = cm_rng_uniform(rng) < options->pstar;
        if (swap_rows) {
            row1 = cm_rng_bounded(rng, base.n_recipients);
            row2 = cm_rng_bounded(rng, base.n_recipients - 1);
            if (row2 >= row1) {
                ++row2;
            }
            slot1 = cm_rng_bounded(rng, options->k);
            slot2 = cm_rng_bounded(rng, options->k);
            old1 = matches[row1 * options->k + slot1];
            old2 = matches[row2 * options->k + slot2];
            log_likelihood_new = -options->beta *
                (base.distance[row1 * base.n_donors + old2] +
                 base.distance[row2 * base.n_donors + old1]);
            log_likelihood_old = -options->beta *
                (base.distance[row1 * base.n_donors + old1] +
                 base.distance[row2 * base.n_donors + old2]);
        } else if (unmatched_count == 0) {
            log_likelihood_new = -INFINITY;
            log_likelihood_old = 0.0;
        } else {
            unmatched_position = cm_rng_bounded(rng, unmatched_count);
            new_donor = unmatched[unmatched_position];
            recipient = cm_rng_bounded(rng, base.n_recipients);
            slot = cm_rng_bounded(rng, options->k);
            old_donor = matches[recipient * options->k + slot];
            log_likelihood_new = -options->beta *
                base.distance[recipient * base.n_donors + new_donor];
            log_likelihood_old = -options->beta *
                base.distance[recipient * base.n_donors + old_donor];
        }
        log_acceptance = fmin(0.0, log_likelihood_new - log_likelihood_old);
        if (cm_log_uniform(rng) < log_acceptance) {
            if (swap_rows) {
                matches[row1 * options->k + slot1] = old2;
                matches[row2 * options->k + slot2] = old1;
                accepted = true;
            } else if (unmatched_count > 0) {
                matches[recipient * options->k + slot] = new_donor;
                unmatched[unmatched_position] = old_donor;
                accepted = true;
            }
        }
        if (accepted) {
            ++acceptance_count;
        }
        timing.mcmc_sampling += cm_timer_now() - started;

        started = cm_timer_now();
        for (r = 0; r < base.n_recipients; ++r) {
            size_t match_slot;
            for (match_slot = 0; match_slot < options->k; ++match_slot) {
                const size_t donor = matches[r * options->k + match_slot];
                matched_outcomes[r * options->k + match_slot] =
                    dataset->Y[base.donors[donor]];
            }
        }
        cm_wls_1tok(recipient_outcomes, base.n_recipients,
                    matched_outcomes, options->k,
                    &series.est[iteration], &series.variance[iteration]);
        /* Deliberately unweighted, matching lm(...) inside Python coeftest. */
        series.cluster[iteration] =
            cm_cluster_se_1tok(recipient_outcomes, matched_outcomes,
                               base.n_recipients, options->k);
        timing.mcmc_estimation += cm_timer_now() - started;
    }

    started = cm_timer_now();
    cm_fill_common_summary(result, &series, options->n_mcmc);
    cm_fill_cluster_summary(result, &series, options->n_mcmc);
    timing.summary += cm_timer_now() - started;

    result->est_med = cm_median_in_place(series.est, options->n_mcmc);
    result->k = options->k;
    result->acceptance_count = acceptance_count;
    result->flags |= CM_RESULT_HAS_K | CM_RESULT_HAS_ACCEPTANCE_COUNT;
    cm_fill_timing(result, options, &timing, (uint64_t)options->n_mcmc);
    status = CM_STATUS_OK;

cleanup:
    free(matches);
    free(unmatched);
    free(recipient_outcomes);
    free(matched_outcomes);
    cm_series_destroy(&series);
    cm_base_destroy(&base);
    return status;
}

void cm_run_options_init(CMRunOptions *options)
{
    if (options == NULL) {
        return;
    }
    options->method = CM_METHOD_CONSTRAINED_1TO1;
    options->n_mcmc = 1000;
    options->beta = 1.0;
    options->k = 2;
    options->pstar = 0.5;
    options->profile = false;
}

void cm_method_result_init(CMMethodResult *result)
{
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
}

const char *cm_method_name(CMMethod method)
{
    static const char *const names[CM_METHOD_COUNT] = {
        "unconstrained_1to1",
        "unconstrained_1tok",
        "unconstrained_1tok_mc",
        "constrained_1to1",
        "constrained_1to1_gibbs",
        "constrained_1tok"
    };

    if ((unsigned int)method >= (unsigned int)CM_METHOD_COUNT) {
        return NULL;
    }
    return names[(size_t)method];
}

bool cm_method_parse(const char *name, CMMethod *method_out)
{
    int method;

    if (name == NULL || method_out == NULL) {
        return false;
    }
    for (method = 0; method < (int)CM_METHOD_COUNT; ++method) {
        if (strcmp(name, cm_method_name((CMMethod)method)) == 0) {
            *method_out = (CMMethod)method;
            return true;
        }
    }
    return false;
}

const char *cm_status_string(CMStatus status)
{
    switch (status) {
        case CM_STATUS_OK:
            return "ok";
        case CM_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case CM_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case CM_STATUS_NUMERICAL_ERROR:
            return "numerical error";
        case CM_STATUS_INTERNAL_ERROR:
            return "internal error";
        default:
            return "unknown status";
    }
}

static CMStatus cm_validate_run(const CMDatasetView *dataset,
                                const CMRunOptions *options,
                                CMRng *rng,
                                CMMethodResult *result,
                                char *error,
                                size_t error_size,
                                size_t *n_recipients_out,
                                size_t *n_donors_out)
{
    size_t n_recipients = 0;
    size_t n_donors = 0;
    size_t i;
    size_t product;
    size_t x_count;

    if (dataset == NULL || options == NULL || rng == NULL || result == NULL) {
        cm_set_error(error, error_size,
                     "dataset, options, rng, and result must all be non-null");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->X == NULL || dataset->Y == NULL || dataset->g == NULL) {
        cm_set_error(error, error_size, "dataset arrays must be non-null");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->n_pop == 0 || dataset->n_cov == 0) {
        cm_set_error(error, error_size,
                     "dataset must contain observations and covariates");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    if (!cm_checked_multiply(dataset->n_pop, dataset->n_cov, &x_count)) {
        cm_set_error(error, error_size, "X dimensions overflow size_t");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    (void)x_count;
    if ((unsigned int)options->method >= (unsigned int)CM_METHOD_COUNT) {
        cm_set_error(error, error_size, "unknown matching method value");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    if (options->n_mcmc == 0) {
        cm_set_error(error, error_size, "n_mcmc must be at least 1");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(options->beta)) {
        cm_set_error(error, error_size, "beta must be finite");
        return CM_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < dataset->n_pop; ++i) {
        size_t covariate;
        if (!isfinite(dataset->Y[i])) {
            cm_set_error(error, error_size, "Y[%zu] is not finite", i);
            return CM_STATUS_INVALID_ARGUMENT;
        }
        if (dataset->g[i] == 1) {
            ++n_recipients;
        } else if (dataset->g[i] == 0) {
            ++n_donors;
        } else {
            cm_set_error(error, error_size,
                         "g[%zu] is %d; group labels must be 0 or 1",
                         i, dataset->g[i]);
            return CM_STATUS_INVALID_ARGUMENT;
        }
        for (covariate = 0; covariate < dataset->n_cov; ++covariate) {
            if (!isfinite(dataset->X[i * dataset->n_cov + covariate])) {
                cm_set_error(error, error_size,
                             "X[%zu,%zu] is not finite", i, covariate);
                return CM_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (n_recipients < 2 || n_donors == 0) {
        cm_set_error(error, error_size,
                     "matching requires at least two recipients and one donor");
        return CM_STATUS_INVALID_ARGUMENT;
    }
    if (!cm_checked_multiply(options->n_mcmc, n_recipients, &product)) {
        cm_set_error(error, error_size, "sampling draw count overflows size_t");
        return CM_STATUS_INVALID_ARGUMENT;
    }

    switch (options->method) {
        case CM_METHOD_UNCONSTRAINED_1TO1:
            break;

        case CM_METHOD_UNCONSTRAINED_1TOK:
        case CM_METHOD_UNCONSTRAINED_1TOK_MC:
            if (options->k < 1 || options->k >= n_donors) {
                cm_set_error(error, error_size,
                             "unconstrained 1-to-k requires 1 <= k < number of donors");
                return CM_STATUS_INVALID_ARGUMENT;
            }
            if (!cm_checked_multiply(n_recipients, options->k, &product)) {
                cm_set_error(error, error_size,
                             "recipient-by-k dimensions overflow");
                return CM_STATUS_INVALID_ARGUMENT;
            }
            break;

        case CM_METHOD_CONSTRAINED_1TO1:
        case CM_METHOD_CONSTRAINED_1TO1_GIBBS:
            if (n_recipients > n_donors) {
                cm_set_error(error, error_size,
                             "constrained 1-to-1 requires at least as many donors as recipients");
                return CM_STATUS_INVALID_ARGUMENT;
            }
            if (!(options->pstar >= 0.0 && options->pstar <= 1.0)) {
                cm_set_error(error, error_size, "pstar must lie in [0, 1]");
                return CM_STATUS_INVALID_ARGUMENT;
            }
            break;

        case CM_METHOD_CONSTRAINED_1TOK:
            if (options->k < 1 ||
                !cm_checked_multiply(n_recipients, options->k, &product) ||
                product > n_donors) {
                cm_set_error(error, error_size,
                             "constrained 1-to-k requires k >= 1 and recipients*k <= donors");
                return CM_STATUS_INVALID_ARGUMENT;
            }
            if (!(options->pstar >= 0.0 && options->pstar <= 1.0)) {
                cm_set_error(error, error_size, "pstar must lie in [0, 1]");
                return CM_STATUS_INVALID_ARGUMENT;
            }
            break;

        case CM_METHOD_COUNT:
        default:
            cm_set_error(error, error_size, "unknown matching method value");
            return CM_STATUS_INVALID_ARGUMENT;
    }

    *n_recipients_out = n_recipients;
    *n_donors_out = n_donors;
    return CM_STATUS_OK;
}

CMStatus cm_run_matching(const CMDatasetView *dataset,
                         const CMRunOptions *options,
                         CMRng *rng,
                         CMMethodResult *result,
                         char *error_message,
                         size_t error_message_size)
{
    CMStatus status;
    size_t n_recipients = 0;
    size_t n_donors = 0;

    if (error_message != NULL && error_message_size > 0) {
        error_message[0] = '\0';
    }
    if (result != NULL) {
        cm_method_result_init(result);
    }
    status = cm_validate_run(dataset, options, rng, result,
                             error_message, error_message_size,
                             &n_recipients, &n_donors);
    if (status != CM_STATUS_OK) {
        return status;
    }
    (void)n_recipients;
    (void)n_donors;
    result->method = options->method;

    switch (options->method) {
        case CM_METHOD_UNCONSTRAINED_1TO1:
            status = cm_run_unconstrained_1to1(
                dataset, options, rng, result,
                error_message, error_message_size);
            break;
        case CM_METHOD_UNCONSTRAINED_1TOK:
            status = cm_run_unconstrained_1tok(
                dataset, options, rng, result,
                error_message, error_message_size);
            break;
        case CM_METHOD_UNCONSTRAINED_1TOK_MC:
            status = cm_run_unconstrained_1tok_mc(
                dataset, options, rng, result,
                error_message, error_message_size);
            break;
        case CM_METHOD_CONSTRAINED_1TO1:
            status = cm_run_constrained_1to1(
                dataset, options, rng, result,
                error_message, error_message_size);
            break;
        case CM_METHOD_CONSTRAINED_1TO1_GIBBS:
            status = cm_run_constrained_1to1_gibbs(
                dataset, options, rng, result,
                error_message, error_message_size);
            break;
        case CM_METHOD_CONSTRAINED_1TOK:
            status = cm_run_constrained_1tok(
                dataset, options, rng, result,
                error_message, error_message_size);
            break;
        case CM_METHOD_COUNT:
        default:
            status = CM_STATUS_INTERNAL_ERROR;
            cm_set_error(error_message, error_message_size,
                         "matching method dispatch failed");
            break;
    }

    if (status != CM_STATUS_OK) {
        cm_method_result_init(result);
    }
    return status;
}
