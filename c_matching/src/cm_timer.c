#include "cm_timer.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

double cm_timer_now(void)
{
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;

    if (!QueryPerformanceCounter(&counter) ||
        !QueryPerformanceFrequency(&frequency) || frequency.QuadPart == 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

#else

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <time.h>

double cm_timer_now(void)
{
    struct timespec ts;

#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
    }
#endif

    if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
    }
    return 0.0;
}

#endif
