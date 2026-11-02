// CUDA test: Burgers equation u_t + (u^2/2)_x = 0, periodic BCs, Rusanov flux.
// Same setup as cpu/*/tests/testBurgers.c, so the results can be compared with the CPU versions.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "include/FVM1DScalarCUDA.cuh"
#include "include/HistoryBinary.h"

// ---------------- Simulation parameters ----------------
static const double xL                = -1.0;   // Left interval extrema
static const double xR                =  1.0;   // Right interval extrema
static const int    N                 = 999999;  // N+1 = 10^6 finite volumes
static const double CFL               = 1.0;    // max|f'(u)| dt / dx (stable for CFL <= 1). dt is computed from it
static const int    T                 = 2001;   // Time steps in the history (final time = dt*(T-1)). Only the recorded rows
                                                // are filled: the GPU holds 2 rows, the host touches only recorded rows
static const int    RECORD_EVERY      = 100;    // Copy back and write every RECORD_EVERY time steps (first and last always)
static const char*  OUTPUT_FILE       = "burgers_cuda.bin";
static const int    THREADS_PER_BLOCK = 256;    // Threads per block
static const int    NUM_BLOCKS        = 3907;   // Blocks: 3907*256 >= 10^6 threads, one per cell. Any value works (grid-stride loop):
                                                // with fewer threads than cells, each thread updates several cells
// -------------------------------------------------------

// Physics, evaluated on one state (usable on the CPU and inside the kernel)
struct Burgers {
    __host__ __device__ static double flux(double u)     { return 0.5*u*u; }
    __host__ __device__ static double fluxGrad(double u) { return u; }
};

static double initialCondition(double x) { return (x < 0.0) ? 1.0 : 0.0; }

int main(void)
{
    double dx = (xR - xL)/(N+1);

    // Initial condition, evaluated at the cell centers
    double* u0 = (double*)malloc(sizeof(double)*(N+1));
    double* u  = (double*)malloc(sizeof(double)*HISTORY_SIZE(T, N));
    for (int j = 0; j <= N; ++j) u0[j] = initialCondition(xL + (j + 0.5)*dx);

    // Time step from the CFL number. The scheme keeps u within the initial range,
    // so max|f'(u0)| bounds the wave speed for the whole run
    double maxSpeed = 0.0;
    for (int j = 0; j <= N; ++j) maxSpeed = fmax(maxSpeed, fabs(Burgers::fluxGrad(u0[j])));
    double dt = CFL*dx/maxSpeed;

    // Create the CUDA context before timing (it takes ~100 ms and is not part of the computation)
    cudaFree(0);

    // Solve and time the whole history computation (end to end: GPU memory, copies and kernels)
    double kernelSeconds;
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DScalarCUDA< Rusanov<Burgers> >(N, u0, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u,
                                         THREADS_PER_BLOCK, NUM_BLOCKS, &kernelSeconds);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms (kernels only: %.3f ms)\n", 1e3*elapsed, 1e3*kernelSeconds);
    printf("Time per cell update: %.3f ns (kernels only: %.3f ns)\n", 1e9*elapsed/((double)(T-1)*(N+1)), 1e9*kernelSeconds/((double)(T-1)*(N+1)));

    writeHistoryBinary(OUTPUT_FILE, u, N, T, xL, xR, dt, RECORD_EVERY);

    free(u0);
    free(u);
    return 0;
}
