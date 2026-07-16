#ifndef CM_MATCHING_H
#define CM_MATCHING_H

#include "cm_rng.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CMMethod {
    CM_METHOD_UNCONSTRAINED_1TO1 = 0,
    CM_METHOD_UNCONSTRAINED_1TOK,
    CM_METHOD_UNCONSTRAINED_1TOK_MC,
    CM_METHOD_CONSTRAINED_1TO1,
    CM_METHOD_CONSTRAINED_1TO1_GIBBS,
    CM_METHOD_CONSTRAINED_1TOK,
    CM_METHOD_COUNT
} CMMethod;

typedef enum CMStatus {
    CM_STATUS_OK = 0,
    CM_STATUS_INVALID_ARGUMENT,
    CM_STATUS_OUT_OF_MEMORY,
    CM_STATUS_NUMERICAL_ERROR,
    CM_STATUS_INTERNAL_ERROR
} CMStatus;

/* All arrays are borrowed and remain owned by the caller.  X is row-major. */
typedef struct CMDatasetView {
    const double *X;
    const double *Y;
    const int *g;
    size_t n_pop;
    size_t n_cov;
} CMDatasetView;

typedef struct CMRunOptions {
    CMMethod method;
    size_t n_mcmc;
    double beta;
    size_t k;
    double pstar;
    bool profile;
} CMRunOptions;

enum {
    CM_RESULT_HAS_K                       = UINT32_C(1) << 0,
    CM_RESULT_HAS_CLUSTER_VARIANCE        = UINT32_C(1) << 1,
    CM_RESULT_HAS_PAIRED_VARIANCE         = UINT32_C(1) << 2,
    CM_RESULT_HAS_ESTDIFF                 = UINT32_C(1) << 3,
    CM_RESULT_HAS_ACCEPTANCE_COUNT        = UINT32_C(1) << 4,
    CM_RESULT_HAS_CHANGE_COUNT            = UINT32_C(1) << 5,
    CM_RESULT_HAS_CHANGED_ITERATION_COUNT = UINT32_C(1) << 6,
    CM_RESULT_HAS_PROFILE                 = UINT32_C(1) << 7
};

typedef struct CMMethodTiming {
    double setup;
    double allocation;
    double mcmc_sampling;
    double mcmc_estimation;
    double summary;

    double mcmc_total;
    double method_total;
    uint64_t n_sampling_draws;
    double time_per_sampling_draw;

    double share_sampling_in_mcmc;
    double share_sampling_in_method;
    double share_estimation_in_method;
    double share_mcmc_in_method;
} CMMethodTiming;

typedef struct CMMethodResult {
    uint32_t flags;
    CMMethod method;

    double est;
    double est_med;
    double wi_var;       /* Python: wiVar */
    double bw_var;       /* Python: bwVar */
    double mi_var;       /* Python: miVar */

    size_t k;
    double wi_var_clus;  /* Python: wiVarclus */
    double mi_var_clus;  /* Python: miVarclus */
    double wi_var_paired;/* Python: wiVarpaired */
    double mi_var_paired;/* Python: miVarpaired */
    double estdiff;

    uint64_t acceptance_count;
    uint64_t change_count;
    uint64_t changed_iteration_count;

    CMMethodTiming timing;
} CMMethodResult;

void cm_run_options_init(CMRunOptions *options);
void cm_method_result_init(CMMethodResult *result);

const char *cm_method_name(CMMethod method);
bool cm_method_parse(const char *name, CMMethod *method_out);
const char *cm_status_string(CMStatus status);

/*
 * Run one dataset replication.  On failure, result is reset and error_message
 * receives a human-readable description when a nonzero buffer is supplied.
 */
CMStatus cm_run_matching(const CMDatasetView *dataset,
                         const CMRunOptions *options,
                         CMRng *rng,
                         CMMethodResult *result,
                         char *error_message,
                         size_t error_message_size);

#ifdef __cplusplus
}
#endif

#endif
