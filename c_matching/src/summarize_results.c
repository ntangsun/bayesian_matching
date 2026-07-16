/*
 * summarize_results.c
 *
 * Dependency-free C17 counterpart to summarize_results.py.  It reads one
 * result CSV (or a glob of result CSVs), prints compact statistical/runtime
 * summaries, and can optionally save one summary row per input file.
 *
 * The CSV reader accepts RFC 4180-style quoted fields, doubled quotes, CRLF,
 * and embedded newlines.  Column order is irrelevant: all inputs are mapped
 * by their header names.
 */

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <sys/stat.h>
#else
#include <glob.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} StringList;

typedef struct {
    char **fields;
    size_t count;
    size_t capacity;
} CsvRecord;

typedef struct {
    size_t n;
    long double sum;
} MeanAccumulator;

typedef struct {
    size_t n;
    long double mean;
    long double m2;
} MomentAccumulator;

typedef struct {
    int source_present;
    int saw_value;
    int mixed;
    char *first_value;
} MetadataValue;

enum MetadataIndex {
    META_METHOD = 0,
    META_BETA,
    META_PSTAR,
    META_N_MCMC,
    META_DATASET_FILE,
    META_COUNT
};

static const char *const metadata_names[META_COUNT] = {
    "method", "beta", "pstar", "n_mcmc", "dataset_file"
};

static const char *const variance_names[] = {
    "wiVar", "bwVar", "miVar", "wiVarclus", "miVarclus",
    "wiVarpaired", "miVarpaired"
};

static const char *const timing_names[] = {
    "time_sim_total",
    "time_sim_prep",
    "time_method_call",
    "time_sim_post",
    "time_setup",
    "time_allocation",
    "time_mcmc_sampling",
    "time_mcmc_estimation",
    "time_mcmc_total",
    "time_summary",
    "time_method_total",
    "share_method_in_sim",
    "share_sampling_in_mcmc",
    "share_mcmc_in_method",
    "n_sampling_draws",
    "time_per_sampling_draw",
    "share_sampling_in_method",
    "share_estimation_in_method"
};

enum TimingIndex {
    TIMING_SIM_TOTAL = 0,
    TIMING_SIM_PREP,
    TIMING_METHOD_CALL,
    TIMING_SIM_POST,
    TIMING_SETUP,
    TIMING_ALLOCATION,
    TIMING_MCMC_SAMPLING,
    TIMING_MCMC_ESTIMATION,
    TIMING_MCMC_TOTAL,
    TIMING_SUMMARY,
    TIMING_METHOD_TOTAL,
    TIMING_SHARE_METHOD_IN_SIM,
    TIMING_SHARE_SAMPLING_IN_MCMC,
    TIMING_SHARE_MCMC_IN_METHOD,
    TIMING_N_SAMPLING_DRAWS,
    TIMING_PER_SAMPLING_DRAW,
    TIMING_SHARE_SAMPLING_IN_METHOD,
    TIMING_SHARE_ESTIMATION_IN_METHOD,
    TIMING_COUNT
};

typedef struct {
    char *file;
    char *path;
    size_t n_sim;

    MetadataValue metadata[META_COUNT];

    int has_est;
    MomentAccumulator est;
    MeanAccumulator abs_bias;

    int variance_present[ARRAY_COUNT(variance_names)];
    MeanAccumulator variance_means[ARRAY_COUNT(variance_names)];

    int has_coverage;
    int has_ci_length;
    MeanAccumulator coverage;
    MeanAccumulator ci_length;

    int has_acceptance_rate;
    int has_acceptance_count;
    MeanAccumulator acceptance_rate;
    MeanAccumulator acceptance_count;

    int has_change_rate;
    int has_change_count;
    MeanAccumulator change_rate;
    MeanAccumulator change_count;

    int has_changed_iteration_rate;
    MeanAccumulator changed_iteration_rate;

    int timing_present[TIMING_COUNT];
    MeanAccumulator timing_means[TIMING_COUNT];
} Summary;

typedef struct {
    const char *results_file;
    const char *results_dir;
    const char *pattern;
    double true_tau;
    int save;
    const char *out;
    const char *out_dir;
    const char *out_prefix;
} Options;

static void die(const char *message)
{
    fprintf(stderr, "error: %s\n", message);
    exit(EXIT_FAILURE);
}

static void die_path(const char *message, const char *path)
{
    fprintf(stderr, "error: %s: %s\n", message, path);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t size)
{
    void *ptr;

    if (size == 0U) {
        size = 1U;
    }
    ptr = malloc(size);
    if (ptr == NULL) {
        die("out of memory");
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size)
{
    void *result;

    if (size == 0U) {
        size = 1U;
    }
    result = realloc(ptr, size);
    if (result == NULL) {
        die("out of memory");
    }
    return result;
}

static char *xstrdup(const char *text)
{
    size_t length = strlen(text);
    char *copy = (char *)xmalloc(length + 1U);
    memcpy(copy, text, length + 1U);
    return copy;
}

static int is_path_separator(char ch)
{
    return ch == '/' || ch == '\\';
}

static char native_separator(void)
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

static char *path_join(const char *left, const char *right)
{
    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    int need_separator = left_len > 0U && !is_path_separator(left[left_len - 1U]);
    char *result = (char *)xmalloc(left_len + right_len + (size_t)need_separator + 1U);

    memcpy(result, left, left_len);
    if (need_separator) {
        result[left_len] = native_separator();
        ++left_len;
    }
    memcpy(result + left_len, right, right_len + 1U);
    return result;
}

static const char *path_basename(const char *path)
{
    const char *base = path;
    const char *cursor;

    for (cursor = path; *cursor != '\0'; ++cursor) {
        if (is_path_separator(*cursor)) {
            base = cursor + 1;
        }
    }
    return base;
}

static char *path_dirname(const char *path)
{
    const char *last = NULL;
    const char *cursor;
    size_t length;
    char *result;

    for (cursor = path; *cursor != '\0'; ++cursor) {
        if (is_path_separator(*cursor)) {
            last = cursor;
        }
    }
    if (last == NULL) {
        return xstrdup(".");
    }
    if (last == path) {
        length = 1U;
    } else if (last == path + 2 && path[1] == ':') {
        length = 3U;
    } else {
        length = (size_t)(last - path);
    }
    result = (char *)xmalloc(length + 1U);
    memcpy(result, path, length);
    result[length] = '\0';
    return result;
}

static int path_is_regular_file(const char *path)
{
#ifdef _WIN32
    DWORD attributes = GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0U;
#else
    struct stat info;
    return stat(path, &info) == 0 && S_ISREG(info.st_mode);
#endif
}

static int make_one_directory(const char *path)
{
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, (mode_t)0777);
#endif
}

static void make_directories(const char *path)
{
    char *copy;
    size_t length;
    size_t start = 0U;
    size_t i;

    if (path == NULL || path[0] == '\0' || strcmp(path, ".") == 0) {
        return;
    }

    copy = xstrdup(path);
    length = strlen(copy);
    while (length > 1U && is_path_separator(copy[length - 1U])) {
        copy[--length] = '\0';
    }

#ifdef _WIN32
    if (length >= 2U && copy[1] == ':') {
        start = 2U;
    } else if (length >= 2U && is_path_separator(copy[0]) &&
               is_path_separator(copy[1])) {
        /* Do not try to create the leading component of a UNC path. */
        start = 2U;
    }
#else
    if (length > 0U && copy[0] == '/') {
        start = 1U;
    }
#endif

    for (i = start; i < length; ++i) {
        if (is_path_separator(copy[i])) {
            char saved = copy[i];
            copy[i] = '\0';
            if (copy[0] != '\0' && make_one_directory(copy) != 0 &&
                errno != EEXIST) {
                free(copy);
                die_path("could not create directory", path);
            }
            copy[i] = saved;
        }
    }
    if (make_one_directory(copy) != 0 && errno != EEXIST) {
        free(copy);
        die_path("could not create directory", path);
    }
    free(copy);
}

static void string_list_push_owned(StringList *list, char *text)
{
    if (list->count == list->capacity) {
        size_t new_capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
        if (new_capacity < list->capacity ||
            new_capacity > SIZE_MAX / sizeof(*list->items)) {
            die("too many paths");
        }
        list->items = (char **)xrealloc(list->items,
                                        new_capacity * sizeof(*list->items));
        list->capacity = new_capacity;
    }
    list->items[list->count++] = text;
}

static void string_list_free(StringList *list)
{
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int compare_strings(const void *left, const void *right)
{
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

static void collect_globbed_paths(const char *directory, const char *pattern,
                                  StringList *paths)
{
    char *search_spec = path_join(directory, pattern);

#ifdef _WIN32
    WIN32_FIND_DATAA data;
    HANDLE handle;
    char *search_directory = path_dirname(search_spec);

    handle = FindFirstFileA(search_spec, &data);
    if (handle != INVALID_HANDLE_VALUE) {
        do {
            if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0U) {
                string_list_push_owned(paths,
                    path_join(search_directory, data.cFileName));
            }
        } while (FindNextFileA(handle, &data) != 0);
        if (GetLastError() != ERROR_NO_MORE_FILES) {
            FindClose(handle);
            free(search_directory);
            free(search_spec);
            die("failed while enumerating result files");
        }
        FindClose(handle);
    } else {
        DWORD error = GetLastError();
        if (error != ERROR_FILE_NOT_FOUND && error != ERROR_PATH_NOT_FOUND) {
            free(search_directory);
            free(search_spec);
            die("could not enumerate result files");
        }
    }
    free(search_directory);
#else
    glob_t matches;
    int status;
    size_t i;

    memset(&matches, 0, sizeof(matches));
    status = glob(search_spec, 0, NULL, &matches);
    if (status != 0 && status != GLOB_NOMATCH) {
        globfree(&matches);
        free(search_spec);
        die("could not enumerate result files");
    }
    for (i = 0U; i < matches.gl_pathc; ++i) {
        if (path_is_regular_file(matches.gl_pathv[i])) {
            string_list_push_owned(paths, xstrdup(matches.gl_pathv[i]));
        }
    }
    globfree(&matches);
#endif

    free(search_spec);
    if (paths->count > 1U) {
        qsort(paths->items, paths->count, sizeof(*paths->items), compare_strings);
    }
}

static void csv_record_push_field(CsvRecord *record, const char *field,
                                  size_t length)
{
    char *copy;

    if (record->count == record->capacity) {
        size_t new_capacity = record->capacity == 0U ? 16U : record->capacity * 2U;
        if (new_capacity < record->capacity ||
            new_capacity > SIZE_MAX / sizeof(*record->fields)) {
            die("CSV record has too many columns");
        }
        record->fields = (char **)xrealloc(record->fields,
            new_capacity * sizeof(*record->fields));
        record->capacity = new_capacity;
    }

    copy = (char *)xmalloc(length + 1U);
    memcpy(copy, field, length);
    copy[length] = '\0';
    record->fields[record->count++] = copy;
}

static void csv_record_clear(CsvRecord *record)
{
    size_t i;
    for (i = 0U; i < record->count; ++i) {
        free(record->fields[i]);
    }
    record->count = 0U;
}

static void csv_record_destroy(CsvRecord *record)
{
    csv_record_clear(record);
    free(record->fields);
    memset(record, 0, sizeof(*record));
}

static void append_character(char **buffer, size_t *length, size_t *capacity,
                             char ch)
{
    if (*length + 1U >= *capacity) {
        size_t new_capacity = *capacity == 0U ? 64U : *capacity * 2U;
        if (new_capacity <= *length) {
            die("CSV field is too large");
        }
        *buffer = (char *)xrealloc(*buffer, new_capacity);
        *capacity = new_capacity;
    }
    (*buffer)[(*length)++] = ch;
}

/* Returns 1 for a record, 0 at clean EOF, and -1 for an unterminated quote. */
static int csv_read_record(FILE *stream, CsvRecord *record)
{
    char *field = NULL;
    size_t field_length = 0U;
    size_t field_capacity = 0U;
    int in_quotes = 0;
    int saw_input = 0;
    int ch;

    csv_record_clear(record);

    while ((ch = fgetc(stream)) != EOF) {
        saw_input = 1;
        if (in_quotes) {
            if (ch == '"') {
                int next = fgetc(stream);
                if (next == '"') {
                    append_character(&field, &field_length, &field_capacity, '"');
                } else {
                    in_quotes = 0;
                    if (next != EOF) {
                        (void)ungetc(next, stream);
                    }
                }
            } else {
                append_character(&field, &field_length, &field_capacity, (char)ch);
            }
            continue;
        }

        if (ch == '"' && field_length == 0U) {
            in_quotes = 1;
        } else if (ch == ',') {
            csv_record_push_field(record, field == NULL ? "" : field,
                                  field_length);
            field_length = 0U;
        } else if (ch == '\n' || ch == '\r') {
            if (ch == '\r') {
                int next = fgetc(stream);
                if (next != '\n' && next != EOF) {
                    (void)ungetc(next, stream);
                }
            }
            csv_record_push_field(record, field == NULL ? "" : field,
                                  field_length);
            free(field);
            return 1;
        } else {
            append_character(&field, &field_length, &field_capacity, (char)ch);
        }
    }

    if (in_quotes) {
        free(field);
        return -1;
    }
    if (!saw_input && field_length == 0U && record->count == 0U) {
        free(field);
        return 0;
    }
    csv_record_push_field(record, field == NULL ? "" : field, field_length);
    free(field);
    return 1;
}

static int csv_record_is_blank(const CsvRecord *record)
{
    return record->count == 1U && record->fields[0][0] == '\0';
}

static void strip_utf8_bom(char *text)
{
    const unsigned char *bytes = (const unsigned char *)text;
    size_t length = strlen(text);
    if (length >= 3U && bytes[0] == 0xEFU && bytes[1] == 0xBBU &&
        bytes[2] == 0xBFU) {
        memmove(text, text + 3, length - 2U);
    }
}

static int header_index(const CsvRecord *header, const char *name)
{
    size_t i;
    for (i = 0U; i < header->count; ++i) {
        if (strcmp(header->fields[i], name) == 0) {
            if (i > (size_t)INT32_MAX) {
                die("CSV has too many columns");
            }
            return (int)i;
        }
    }
    return -1;
}

static const char *record_value(const CsvRecord *record, int index)
{
    if (index < 0 || (size_t)index >= record->count) {
        return "";
    }
    return record->fields[(size_t)index];
}

static int ascii_equal_ignore_case(const char *left, const char *right)
{
    while (*left != '\0' && *right != '\0') {
        unsigned char a = (unsigned char)*left++;
        unsigned char b = (unsigned char)*right++;
        if (tolower(a) != tolower(b)) {
            return 0;
        }
    }
    return *left == '\0' && *right == '\0';
}

static void trimmed_bounds(const char *text, const char **begin, const char **end)
{
    const char *first = text;
    const char *last = text + strlen(text);

    while (first < last && isspace((unsigned char)*first)) {
        ++first;
    }
    while (last > first && isspace((unsigned char)last[-1])) {
        --last;
    }
    *begin = first;
    *end = last;
}

static char *trimmed_copy(const char *text)
{
    const char *begin;
    const char *end;
    size_t length;
    char *copy;

    trimmed_bounds(text, &begin, &end);
    length = (size_t)(end - begin);
    copy = (char *)xmalloc(length + 1U);
    memcpy(copy, begin, length);
    copy[length] = '\0';
    return copy;
}

static int text_is_missing(const char *text)
{
    char *copy = trimmed_copy(text);
    int missing = copy[0] == '\0' ||
                  ascii_equal_ignore_case(copy, "nan") ||
                  ascii_equal_ignore_case(copy, "na") ||
                  ascii_equal_ignore_case(copy, "n/a") ||
                  ascii_equal_ignore_case(copy, "null") ||
                  ascii_equal_ignore_case(copy, "none");
    free(copy);
    return missing;
}

static int parse_number(const char *text, double *value)
{
    char *copy;
    char *end;
    double parsed;

    if (text_is_missing(text)) {
        return 0;
    }
    copy = trimmed_copy(text);
    errno = 0;
    parsed = strtod(copy, &end);
    if (end == copy || *end != '\0' || isnan(parsed)) {
        free(copy);
        return 0;
    }
    *value = parsed;
    free(copy);
    return 1;
}

static int metadata_values_equal(const char *left, const char *right)
{
    double a;
    double b;
    if (parse_number(left, &a) && parse_number(right, &b)) {
        return a == b;
    }
    return strcmp(left, right) == 0;
}

static void metadata_add(MetadataValue *metadata, const char *text)
{
    char *copy;

    if (text_is_missing(text)) {
        return;
    }
    copy = trimmed_copy(text);
    if (!metadata->saw_value) {
        metadata->first_value = copy;
        metadata->saw_value = 1;
    } else {
        if (!metadata_values_equal(metadata->first_value, copy)) {
            metadata->mixed = 1;
        }
        free(copy);
    }
}

static void mean_add(MeanAccumulator *accumulator, double value)
{
    if (isnan(value)) {
        return;
    }
    accumulator->sum += (long double)value;
    ++accumulator->n;
}

static double mean_value(const MeanAccumulator *accumulator)
{
    if (accumulator->n == 0U) {
        return NAN;
    }
    return (double)(accumulator->sum / (long double)accumulator->n);
}

static void moment_add(MomentAccumulator *accumulator, double value)
{
    long double delta;
    long double delta2;

    if (isnan(value)) {
        return;
    }
    ++accumulator->n;
    delta = (long double)value - accumulator->mean;
    accumulator->mean += delta / (long double)accumulator->n;
    delta2 = (long double)value - accumulator->mean;
    accumulator->m2 += delta * delta2;
}

static double moment_mean(const MomentAccumulator *accumulator)
{
    return accumulator->n == 0U ? NAN : (double)accumulator->mean;
}

static double moment_sample_variance(const MomentAccumulator *accumulator)
{
    if (accumulator->n < 2U) {
        return NAN;
    }
    return (double)(accumulator->m2 / (long double)(accumulator->n - 1U));
}

static double safe_ratio(double numerator, double denominator)
{
    if (denominator != 0.0) {
        return numerator / denominator;
    }
    if (numerator == 0.0) {
        return NAN;
    }
    return copysign(INFINITY, numerator * denominator);
}

static void summary_destroy(Summary *summary)
{
    size_t i;
    free(summary->file);
    free(summary->path);
    for (i = 0U; i < META_COUNT; ++i) {
        free(summary->metadata[i].first_value);
    }
    memset(summary, 0, sizeof(*summary));
}

static void summarize_file(const char *path, double true_tau, Summary *summary)
{
    FILE *stream;
    CsvRecord header = {0};
    CsvRecord row = {0};
    int read_status;
    int metadata_columns[META_COUNT];
    int variance_columns[ARRAY_COUNT(variance_names)];
    int timing_columns[TIMING_COUNT];
    int est_column;
    int ll_column;
    int ul_column;
    int covers_true_column;
    int acceptance_count_column;
    int n_mcmc_column;
    int change_count_column;
    int n_sampling_draws_column;
    int changed_iteration_count_column;
    size_t i;

    memset(summary, 0, sizeof(*summary));
    summary->path = xstrdup(path);
    summary->file = xstrdup(path_basename(path));

    stream = fopen(path, "rb");
    if (stream == NULL) {
        die_path("could not open result CSV", path);
    }
    read_status = csv_read_record(stream, &header);
    if (read_status <= 0 || csv_record_is_blank(&header)) {
        fclose(stream);
        csv_record_destroy(&header);
        die_path(read_status < 0 ? "unterminated quote in CSV header" :
                                  "result CSV is empty", path);
    }
    strip_utf8_bom(header.fields[0]);

    for (i = 0U; i < META_COUNT; ++i) {
        metadata_columns[i] = header_index(&header, metadata_names[i]);
        summary->metadata[i].source_present = metadata_columns[i] >= 0;
    }
    for (i = 0U; i < ARRAY_COUNT(variance_names); ++i) {
        variance_columns[i] = header_index(&header, variance_names[i]);
        summary->variance_present[i] = variance_columns[i] >= 0;
    }
    for (i = 0U; i < TIMING_COUNT; ++i) {
        timing_columns[i] = header_index(&header, timing_names[i]);
        summary->timing_present[i] = timing_columns[i] >= 0;
    }

    est_column = header_index(&header, "est");
    summary->has_est = est_column >= 0;
    ll_column = header_index(&header, "ll");
    ul_column = header_index(&header, "ul");
    covers_true_column = header_index(&header, "covers_true");
    summary->has_coverage = (ll_column >= 0 && ul_column >= 0) ||
                            covers_true_column >= 0;
    summary->has_ci_length = ll_column >= 0 && ul_column >= 0;

    acceptance_count_column = header_index(&header, "acceptance_count");
    n_mcmc_column = header_index(&header, "n_mcmc");
    summary->has_acceptance_rate = acceptance_count_column >= 0 &&
                                   n_mcmc_column >= 0;
    summary->has_acceptance_count = acceptance_count_column >= 0 &&
                                    n_mcmc_column < 0;

    change_count_column = header_index(&header, "change_count");
    n_sampling_draws_column = header_index(&header, "n_sampling_draws");
    summary->has_change_rate = change_count_column >= 0 &&
                               n_sampling_draws_column >= 0;
    summary->has_change_count = change_count_column >= 0 &&
                                n_sampling_draws_column < 0;

    changed_iteration_count_column =
        header_index(&header, "changed_iteration_count");
    summary->has_changed_iteration_rate =
        changed_iteration_count_column >= 0 && n_mcmc_column >= 0;

    while ((read_status = csv_read_record(stream, &row)) > 0) {
        double value;
        double other;

        if (csv_record_is_blank(&row)) {
            continue;
        }
        ++summary->n_sim;

        for (i = 0U; i < META_COUNT; ++i) {
            if (metadata_columns[i] >= 0) {
                metadata_add(&summary->metadata[i],
                             record_value(&row, metadata_columns[i]));
            }
        }

        if (est_column >= 0 &&
            parse_number(record_value(&row, est_column), &value)) {
            moment_add(&summary->est, value);
            mean_add(&summary->abs_bias, fabs(value - true_tau));
        }

        for (i = 0U; i < ARRAY_COUNT(variance_names); ++i) {
            if (variance_columns[i] >= 0 &&
                parse_number(record_value(&row, variance_columns[i]), &value)) {
                mean_add(&summary->variance_means[i], value);
            }
        }

        if (ll_column >= 0 && ul_column >= 0) {
            int ll_valid = parse_number(record_value(&row, ll_column), &value);
            int ul_valid = parse_number(record_value(&row, ul_column), &other);
            mean_add(&summary->coverage,
                     (ll_valid && ul_valid && value <= true_tau &&
                      true_tau <= other) ? 1.0 : 0.0);
            if (ll_valid && ul_valid) {
                mean_add(&summary->ci_length, other - value);
            }
        } else if (covers_true_column >= 0 &&
                   parse_number(record_value(&row, covers_true_column), &value)) {
            mean_add(&summary->coverage, value);
        }

        if (summary->has_acceptance_rate &&
            parse_number(record_value(&row, acceptance_count_column), &value) &&
            parse_number(record_value(&row, n_mcmc_column), &other)) {
            mean_add(&summary->acceptance_rate, safe_ratio(value, other));
        } else if (summary->has_acceptance_count &&
                   parse_number(record_value(&row, acceptance_count_column),
                                &value)) {
            mean_add(&summary->acceptance_count, value);
        }

        if (summary->has_change_rate &&
            parse_number(record_value(&row, change_count_column), &value) &&
            parse_number(record_value(&row, n_sampling_draws_column), &other)) {
            mean_add(&summary->change_rate, safe_ratio(value, other));
        } else if (summary->has_change_count &&
                   parse_number(record_value(&row, change_count_column), &value)) {
            mean_add(&summary->change_count, value);
        }

        if (summary->has_changed_iteration_rate &&
            parse_number(record_value(&row, changed_iteration_count_column),
                         &value) &&
            parse_number(record_value(&row, n_mcmc_column), &other)) {
            mean_add(&summary->changed_iteration_rate, safe_ratio(value, other));
        }

        for (i = 0U; i < TIMING_COUNT; ++i) {
            if (timing_columns[i] >= 0 &&
                parse_number(record_value(&row, timing_columns[i]), &value)) {
                mean_add(&summary->timing_means[i], value);
            }
        }
    }

    if (read_status < 0) {
        fclose(stream);
        csv_record_destroy(&row);
        csv_record_destroy(&header);
        die_path("unterminated quoted field in result CSV", path);
    }
    if (ferror(stream)) {
        fclose(stream);
        csv_record_destroy(&row);
        csv_record_destroy(&header);
        die_path("could not read result CSV", path);
    }

    fclose(stream);
    csv_record_destroy(&row);
    csv_record_destroy(&header);
}

static int summaries_have_metadata(const Summary *summaries, size_t count,
                                   size_t index)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (summaries[i].metadata[index].source_present) {
            return 1;
        }
    }
    return 0;
}

static int summaries_have_variance(const Summary *summaries, size_t count,
                                   size_t index)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (summaries[i].variance_present[index]) {
            return 1;
        }
    }
    return 0;
}

static int summaries_have_timing(const Summary *summaries, size_t count,
                                 size_t index)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (summaries[i].timing_present[index]) {
            return 1;
        }
    }
    return 0;
}

#define DEFINE_ANY_FLAG(function_name, field_name)                         \
    static int function_name(const Summary *summaries, size_t count)       \
    {                                                                      \
        size_t i;                                                          \
        for (i = 0U; i < count; ++i) {                                    \
            if (summaries[i].field_name) {                                \
                return 1;                                                  \
            }                                                              \
        }                                                                  \
        return 0;                                                          \
    }

DEFINE_ANY_FLAG(any_has_est, has_est)
DEFINE_ANY_FLAG(any_has_coverage, has_coverage)
DEFINE_ANY_FLAG(any_has_ci_length, has_ci_length)
DEFINE_ANY_FLAG(any_has_acceptance_rate, has_acceptance_rate)
DEFINE_ANY_FLAG(any_has_acceptance_count, has_acceptance_count)
DEFINE_ANY_FLAG(any_has_change_rate, has_change_rate)
DEFINE_ANY_FLAG(any_has_change_count, has_change_count)
DEFINE_ANY_FLAG(any_has_changed_iteration_rate, has_changed_iteration_rate)

#undef DEFINE_ANY_FLAG

static void csv_write_text(FILE *stream, const char *text)
{
    const char *cursor;
    int quote = 0;

    for (cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor == ',' || *cursor == '"' || *cursor == '\r' ||
            *cursor == '\n') {
            quote = 1;
            break;
        }
    }
    if (!quote) {
        fputs(text, stream);
        return;
    }
    fputc('"', stream);
    for (cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor == '"') {
            fputc('"', stream);
        }
        fputc(*cursor, stream);
    }
    fputc('"', stream);
}

static void csv_write_number(FILE *stream, double value)
{
    if (!isnan(value)) {
        fprintf(stream, "%.17g", value);
    }
}

static void csv_next_field(FILE *stream, int *first)
{
    if (!*first) {
        fputc(',', stream);
    }
    *first = 0;
}

static void csv_header_field(FILE *stream, int *first, const char *name)
{
    csv_next_field(stream, first);
    csv_write_text(stream, name);
}

static void csv_text_field(FILE *stream, int *first, const char *text)
{
    csv_next_field(stream, first);
    csv_write_text(stream, text);
}

static void csv_number_field(FILE *stream, int *first, double value)
{
    csv_next_field(stream, first);
    csv_write_number(stream, value);
}

static void csv_empty_field(FILE *stream, int *first)
{
    csv_next_field(stream, first);
}

static void csv_metadata_field(FILE *stream, int *first,
                               const MetadataValue *metadata,
                               size_t metadata_index)
{
    double numeric;

    csv_next_field(stream, first);
    if (!metadata->source_present) {
        return;
    }
    if (!metadata->saw_value || metadata->mixed) {
        csv_write_text(stream, "mixed");
        return;
    }
    if ((metadata_index == META_BETA || metadata_index == META_PSTAR ||
         metadata_index == META_N_MCMC) &&
        parse_number(metadata->first_value, &numeric)) {
        csv_write_number(stream, numeric);
    } else {
        csv_write_text(stream, metadata->first_value);
    }
}

static void write_summary_csv(const char *path, const Summary *summaries,
                              size_t count, double true_tau)
{
    FILE *stream;
    int first = 1;
    size_t i;
    size_t j;
    int include_est = any_has_est(summaries, count);
    int include_coverage = any_has_coverage(summaries, count);
    int include_ci_length = any_has_ci_length(summaries, count);
    int include_acceptance_rate = any_has_acceptance_rate(summaries, count);
    int include_acceptance_count = any_has_acceptance_count(summaries, count);
    int include_change_rate = any_has_change_rate(summaries, count);
    int include_change_count = any_has_change_count(summaries, count);
    int include_changed_iteration_rate =
        any_has_changed_iteration_rate(summaries, count);

    stream = fopen(path, "wb");
    if (stream == NULL) {
        die_path("could not create summary CSV", path);
    }

    csv_header_field(stream, &first, "file");
    csv_header_field(stream, &first, "path");
    csv_header_field(stream, &first, "n_sim");
    for (j = 0U; j < META_COUNT; ++j) {
        if (summaries_have_metadata(summaries, count, j)) {
            csv_header_field(stream, &first, metadata_names[j]);
        }
    }
    if (include_est) {
        csv_header_field(stream, &first, "mean_est");
        csv_header_field(stream, &first, "bias");
        csv_header_field(stream, &first, "abs_bias");
        csv_header_field(stream, &first, "empirical_var_est");
    }
    for (j = 0U; j < ARRAY_COUNT(variance_names); ++j) {
        if (summaries_have_variance(summaries, count, j)) {
            char name[64];
            (void)snprintf(name, sizeof(name), "mean_%s", variance_names[j]);
            csv_header_field(stream, &first, name);
        }
    }
    if (include_coverage) {
        csv_header_field(stream, &first, "coverage");
    }
    if (include_ci_length) {
        csv_header_field(stream, &first, "mean_ci_length");
    }
    if (include_acceptance_rate) {
        csv_header_field(stream, &first, "mean_acceptance_rate");
    }
    if (include_acceptance_count) {
        csv_header_field(stream, &first, "mean_acceptance_count");
    }
    if (include_change_rate) {
        csv_header_field(stream, &first, "mean_change_rate");
    }
    if (include_change_count) {
        csv_header_field(stream, &first, "mean_change_count");
    }
    if (include_changed_iteration_rate) {
        csv_header_field(stream, &first, "mean_changed_iteration_rate");
    }
    for (j = 0U; j < TIMING_COUNT; ++j) {
        if (summaries_have_timing(summaries, count, j)) {
            char name[96];
            (void)snprintf(name, sizeof(name), "mean_%s", timing_names[j]);
            csv_header_field(stream, &first, name);
        }
    }
    fputc('\n', stream);

    for (i = 0U; i < count; ++i) {
        const Summary *summary = &summaries[i];
        double estimate_mean = moment_mean(&summary->est);
        first = 1;
        csv_text_field(stream, &first, summary->file);
        csv_text_field(stream, &first, summary->path);
        csv_next_field(stream, &first);
        fprintf(stream, "%zu", summary->n_sim);

        for (j = 0U; j < META_COUNT; ++j) {
            if (summaries_have_metadata(summaries, count, j)) {
                csv_metadata_field(stream, &first, &summary->metadata[j], j);
            }
        }
        if (include_est) {
            if (summary->has_est) {
                csv_number_field(stream, &first, estimate_mean);
                csv_number_field(stream, &first, estimate_mean - true_tau);
                csv_number_field(stream, &first,
                                 mean_value(&summary->abs_bias));
                csv_number_field(stream, &first,
                                 moment_sample_variance(&summary->est));
            } else {
                csv_empty_field(stream, &first);
                csv_empty_field(stream, &first);
                csv_empty_field(stream, &first);
                csv_empty_field(stream, &first);
            }
        }
        for (j = 0U; j < ARRAY_COUNT(variance_names); ++j) {
            if (summaries_have_variance(summaries, count, j)) {
                if (summary->variance_present[j]) {
                    csv_number_field(stream, &first,
                                     mean_value(&summary->variance_means[j]));
                } else {
                    csv_empty_field(stream, &first);
                }
            }
        }
        if (include_coverage) {
            if (summary->has_coverage) {
                csv_number_field(stream, &first,
                                 mean_value(&summary->coverage));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        if (include_ci_length) {
            if (summary->has_ci_length) {
                csv_number_field(stream, &first,
                                 mean_value(&summary->ci_length));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        if (include_acceptance_rate) {
            if (summary->has_acceptance_rate) {
                csv_number_field(stream, &first,
                                 mean_value(&summary->acceptance_rate));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        if (include_acceptance_count) {
            if (summary->has_acceptance_count) {
                csv_number_field(stream, &first,
                                 mean_value(&summary->acceptance_count));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        if (include_change_rate) {
            if (summary->has_change_rate) {
                csv_number_field(stream, &first,
                                 mean_value(&summary->change_rate));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        if (include_change_count) {
            if (summary->has_change_count) {
                csv_number_field(stream, &first,
                                 mean_value(&summary->change_count));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        if (include_changed_iteration_rate) {
            if (summary->has_changed_iteration_rate) {
                csv_number_field(stream, &first,
                    mean_value(&summary->changed_iteration_rate));
            } else {
                csv_empty_field(stream, &first);
            }
        }
        for (j = 0U; j < TIMING_COUNT; ++j) {
            if (summaries_have_timing(summaries, count, j)) {
                if (summary->timing_present[j]) {
                    csv_number_field(stream, &first,
                                     mean_value(&summary->timing_means[j]));
                } else {
                    csv_empty_field(stream, &first);
                }
            }
        }
        fputc('\n', stream);
    }

    if (fclose(stream) != 0) {
        die_path("could not finish writing summary CSV", path);
    }
}

static void print_separator(void)
{
    puts("================================================================================");
}

static const char *metadata_display(const MetadataValue *metadata)
{
    if (!metadata->saw_value || metadata->mixed) {
        return "mixed";
    }
    return metadata->first_value;
}

static void print_statistical_summary(const Summary *summaries, size_t count,
                                      double true_tau)
{
    size_t i;
    int show_method = summaries_have_metadata(summaries, count, META_METHOD);
    int show_beta = summaries_have_metadata(summaries, count, META_BETA);
    int show_est = any_has_est(summaries, count);
    int show_mivar = 0;
    int show_coverage = any_has_coverage(summaries, count);
    int show_ci = any_has_ci_length(summaries, count);
    int show_acceptance = any_has_acceptance_rate(summaries, count);
    int show_change = any_has_change_rate(summaries, count);
    int show_changed_iteration =
        any_has_changed_iteration_rate(summaries, count);

    for (i = 0U; i < ARRAY_COUNT(variance_names); ++i) {
        if (strcmp(variance_names[i], "miVar") == 0) {
            show_mivar = summaries_have_variance(summaries, count, i);
            break;
        }
    }

    puts("\n=== Statistical Summary ===");
    printf("file");
    if (show_method) printf(" | method");
    if (show_beta) printf(" | beta");
    printf(" | n_sim");
    if (show_est) printf(" | mean_est | bias | abs_bias");
    if (show_mivar) printf(" | mean_miVar");
    if (show_coverage) printf(" | coverage");
    if (show_ci) printf(" | mean_ci_length");
    if (show_acceptance) printf(" | mean_acceptance_rate");
    if (show_change) printf(" | mean_change_rate");
    if (show_changed_iteration) printf(" | mean_changed_iteration_rate");
    putchar('\n');

    for (i = 0U; i < count; ++i) {
        const Summary *summary = &summaries[i];
        size_t j;
        printf("%s", summary->file);
        if (show_method) {
            printf(" | %s", summary->metadata[META_METHOD].source_present ?
                metadata_display(&summary->metadata[META_METHOD]) : "");
        }
        if (show_beta) {
            printf(" | %s", summary->metadata[META_BETA].source_present ?
                metadata_display(&summary->metadata[META_BETA]) : "");
        }
        printf(" | %zu", summary->n_sim);
        if (show_est) {
            if (summary->has_est) {
                double estimate = moment_mean(&summary->est);
                printf(" | %.17g | %.17g | %.17g", estimate,
                       estimate - true_tau, mean_value(&summary->abs_bias));
            } else {
                printf(" |  |  | ");
            }
        }
        if (show_mivar) {
            int printed = 0;
            for (j = 0U; j < ARRAY_COUNT(variance_names); ++j) {
                if (strcmp(variance_names[j], "miVar") == 0) {
                    if (summary->variance_present[j]) {
                        printf(" | %.17g",
                               mean_value(&summary->variance_means[j]));
                    } else {
                        printf(" | ");
                    }
                    printed = 1;
                    break;
                }
            }
            if (!printed) {
                printf(" | ");
            }
        }
        if (show_coverage) {
            if (summary->has_coverage) {
                printf(" | %.17g", mean_value(&summary->coverage));
            } else {
                printf(" | ");
            }
        }
        if (show_ci) {
            if (summary->has_ci_length) {
                printf(" | %.17g", mean_value(&summary->ci_length));
            } else {
                printf(" | ");
            }
        }
        if (show_acceptance) {
            if (summary->has_acceptance_rate) {
                printf(" | %.17g", mean_value(&summary->acceptance_rate));
            } else {
                printf(" | ");
            }
        }
        if (show_change) {
            if (summary->has_change_rate) {
                printf(" | %.17g", mean_value(&summary->change_rate));
            } else {
                printf(" | ");
            }
        }
        if (show_changed_iteration) {
            if (summary->has_changed_iteration_rate) {
                printf(" | %.17g",
                       mean_value(&summary->changed_iteration_rate));
            } else {
                printf(" | ");
            }
        }
        putchar('\n');
    }
}

static void print_runtime_value(const Summary *summary, size_t index)
{
    if (summary->timing_present[index]) {
        double value = mean_value(&summary->timing_means[index]);
        char label[96];
        if (isnan(value)) {
            return;
        }
        (void)snprintf(label, sizeof(label), "mean_%s", timing_names[index]);
        printf("%-35s %.17g\n", label, value);
    }
}

static int summary_has_any_timing(const Summary *summary)
{
    size_t i;
    for (i = 0U; i < TIMING_COUNT; ++i) {
        if (summary->timing_present[i]) {
            return 1;
        }
    }
    return 0;
}

static void print_runtime_summaries(const Summary *summaries, size_t count)
{
    static const size_t simulation_loop[] = {
        TIMING_SIM_TOTAL, TIMING_SIM_PREP, TIMING_METHOD_CALL,
        TIMING_SIM_POST, TIMING_SHARE_METHOD_IN_SIM
    };
    static const size_t method_internals[] = {
        TIMING_METHOD_TOTAL, TIMING_SETUP, TIMING_ALLOCATION, TIMING_SUMMARY,
        TIMING_MCMC_TOTAL, TIMING_MCMC_SAMPLING, TIMING_MCMC_ESTIMATION
    };
    static const size_t mcmc_breakdown[] = {
        TIMING_N_SAMPLING_DRAWS, TIMING_PER_SAMPLING_DRAW,
        TIMING_SHARE_SAMPLING_IN_MCMC, TIMING_SHARE_SAMPLING_IN_METHOD,
        TIMING_SHARE_ESTIMATION_IN_METHOD, TIMING_SHARE_MCMC_IN_METHOD
    };
    size_t i;
    size_t j;

    for (i = 0U; i < count; ++i) {
        const Summary *summary = &summaries[i];
        if (!summary_has_any_timing(summary)) {
            continue;
        }
        putchar('\n');
        print_separator();
        printf("Runtime Profile: %s\n", summary->file);
        print_separator();

        puts("\n[Simulation Loop]");
        for (j = 0U; j < ARRAY_COUNT(simulation_loop); ++j) {
            print_runtime_value(summary, simulation_loop[j]);
        }

        puts("\n[Method Internals]");
        for (j = 0U; j < ARRAY_COUNT(method_internals); ++j) {
            print_runtime_value(summary, method_internals[j]);
        }

        puts("\n[MCMC Breakdown]");
        for (j = 0U; j < ARRAY_COUNT(mcmc_breakdown); ++j) {
            print_runtime_value(summary, mcmc_breakdown[j]);
        }
    }
}

static void print_usage(FILE *stream, const char *program)
{
    fprintf(stream,
        "Usage:\n"
        "  %s --results-file FILE [options]\n"
        "  %s --results-dir DIR [--pattern GLOB] [options]\n\n"
        "Options:\n"
        "  --results-file FILE  Summarize one result CSV.\n"
        "  --results-dir DIR    Directory containing result CSVs.\n"
        "  --pattern GLOB       File pattern (default: *.csv).\n"
        "  --true-tau VALUE     True treatment effect (default: 1).\n"
        "  --save               Save the full summary CSV.\n"
        "  --out FILE           Exact output path (with --save).\n"
        "  --out-dir DIR        Auto-output directory (default: summaries).\n"
        "  --out-prefix PREFIX  Auto-output prefix (default: summary).\n"
        "  -h, --help           Show this help.\n",
        program, program);
}

static const char *option_value(int argc, char **argv, int *index,
                                const char *option, const char *inline_value)
{
    if (inline_value != NULL) {
        if (*inline_value == '\0') {
            fprintf(stderr, "error: %s requires a value\n", option);
            exit(EXIT_FAILURE);
        }
        return inline_value;
    }
    if (*index + 1 >= argc) {
        fprintf(stderr, "error: %s requires a value\n", option);
        exit(EXIT_FAILURE);
    }
    ++*index;
    return argv[*index];
}

static void parse_options(int argc, char **argv, Options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->pattern = "*.csv";
    options->true_tau = 1.0;
    options->out_dir = "summaries";
    options->out_prefix = "summary";

    for (i = 1; i < argc; ++i) {
        const char *argument = argv[i];
        const char *equals = strchr(argument, '=');
        size_t name_length = equals == NULL ? strlen(argument) :
                                             (size_t)(equals - argument);
        const char *inline_value = equals == NULL ? NULL : equals + 1;

#define OPTION_IS(name) (name_length == strlen(name) &&                 \
                         strncmp(argument, name, name_length) == 0)
        if (OPTION_IS("--results-file")) {
            options->results_file = option_value(argc, argv, &i,
                "--results-file", inline_value);
        } else if (OPTION_IS("--results-dir")) {
            options->results_dir = option_value(argc, argv, &i,
                "--results-dir", inline_value);
        } else if (OPTION_IS("--pattern")) {
            options->pattern = option_value(argc, argv, &i,
                "--pattern", inline_value);
        } else if (OPTION_IS("--true-tau")) {
            const char *text = option_value(argc, argv, &i,
                "--true-tau", inline_value);
            if (!parse_number(text, &options->true_tau)) {
                fprintf(stderr, "error: invalid --true-tau value: %s\n", text);
                exit(EXIT_FAILURE);
            }
        } else if (OPTION_IS("--save")) {
            if (inline_value != NULL) {
                die("--save does not accept a value");
            }
            options->save = 1;
        } else if (OPTION_IS("--out")) {
            options->out = option_value(argc, argv, &i, "--out", inline_value);
        } else if (OPTION_IS("--out-dir")) {
            options->out_dir = option_value(argc, argv, &i,
                "--out-dir", inline_value);
        } else if (OPTION_IS("--out-prefix")) {
            options->out_prefix = option_value(argc, argv, &i,
                "--out-prefix", inline_value);
        } else if ((strcmp(argument, "--help") == 0 ||
                    strcmp(argument, "-h") == 0) && equals == NULL) {
            print_usage(stdout, argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argument);
            print_usage(stderr, argv[0]);
            exit(EXIT_FAILURE);
        }
#undef OPTION_IS
    }

    if ((options->results_file == NULL) == (options->results_dir == NULL)) {
        die("provide exactly one of --results-file or --results-dir");
    }
}

static int parse_incremented_name(const char *filename, const char *prefix,
                                  size_t *identifier)
{
    size_t prefix_length = strlen(prefix);
    const char *digits;
    const char *cursor;
    size_t value = 0U;

    if (strncmp(filename, prefix, prefix_length) != 0 ||
        filename[prefix_length] != '_') {
        return 0;
    }
    digits = filename + prefix_length + 1U;
    if (!isdigit((unsigned char)*digits)) {
        return 0;
    }
    cursor = digits;
    while (isdigit((unsigned char)*cursor)) {
        unsigned int digit = (unsigned int)(*cursor - '0');
        if (value > (SIZE_MAX - digit) / 10U) {
            return 0;
        }
        value = value * 10U + digit;
        ++cursor;
    }
    if (strcmp(cursor, ".csv") != 0) {
        return 0;
    }
    *identifier = value;
    return 1;
}

static char *next_incremented_path(const char *out_dir, const char *prefix)
{
    StringList matches = {0};
    char *pattern;
    size_t pattern_length = strlen(prefix) + strlen("_*.csv") + 1U;
    size_t max_identifier = 0U;
    size_t i;
    size_t next_identifier;
    int required;
    char *filename;
    char *result;

    make_directories(out_dir);
    pattern = (char *)xmalloc(pattern_length);
    (void)snprintf(pattern, pattern_length, "%s_*.csv", prefix);
    collect_globbed_paths(out_dir, pattern, &matches);
    free(pattern);

    for (i = 0U; i < matches.count; ++i) {
        size_t identifier;
        if (parse_incremented_name(path_basename(matches.items[i]), prefix,
                                   &identifier) && identifier > max_identifier) {
            max_identifier = identifier;
        }
    }
    string_list_free(&matches);
    if (max_identifier == SIZE_MAX) {
        die("summary filename counter overflow");
    }
    next_identifier = max_identifier + 1U;

    required = snprintf(NULL, 0, "%s_%zu.csv", prefix, next_identifier);
    if (required < 0) {
        die("could not format summary filename");
    }
    filename = (char *)xmalloc((size_t)required + 1U);
    (void)snprintf(filename, (size_t)required + 1U, "%s_%zu.csv", prefix,
                   next_identifier);
    result = path_join(out_dir, filename);
    free(filename);
    return result;
}

int main(int argc, char **argv)
{
    Options options;
    StringList paths = {0};
    Summary *summaries;
    size_t i;
    char *output_path = NULL;
    char *output_directory = NULL;

    (void)setlocale(LC_NUMERIC, "C");
    parse_options(argc, argv, &options);

    if (options.results_file != NULL) {
        if (!path_is_regular_file(options.results_file)) {
            die_path("result file does not exist", options.results_file);
        }
        string_list_push_owned(&paths, xstrdup(options.results_file));
    } else {
        collect_globbed_paths(options.results_dir, options.pattern, &paths);
        if (paths.count == 0U) {
            die("no result CSV files found");
        }
    }

    if (paths.count > SIZE_MAX / sizeof(*summaries)) {
        die("too many result files");
    }
    summaries = (Summary *)xmalloc(paths.count * sizeof(*summaries));
    memset(summaries, 0, paths.count * sizeof(*summaries));
    for (i = 0U; i < paths.count; ++i) {
        summarize_file(paths.items[i], options.true_tau, &summaries[i]);
    }

    print_statistical_summary(summaries, paths.count, options.true_tau);
    print_runtime_summaries(summaries, paths.count);

    if (options.save) {
        if (options.out != NULL) {
            output_path = xstrdup(options.out);
            output_directory = path_dirname(output_path);
            make_directories(output_directory);
        } else {
            output_path = next_incremented_path(options.out_dir,
                                                options.out_prefix);
        }
        write_summary_csv(output_path, summaries, paths.count,
                          options.true_tau);
        printf("\nFull summary saved to: %s\n", output_path);
    } else {
        puts("\nSummary was not saved. Add --save to save it.");
    }

    free(output_directory);
    free(output_path);
    for (i = 0U; i < paths.count; ++i) {
        summary_destroy(&summaries[i]);
    }
    free(summaries);
    string_list_free(&paths);
    return EXIT_SUCCESS;
}
