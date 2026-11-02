// Test: linear transport equation u_t + a u_x = 0, periodic BCs, Rusanov flux.
// Initial condition: u = 1 if x < 0, u = 0 if x > 0.
// Exact solution: u(x, t) = u_0(x - a t), wrapped periodically into [xL, xR).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DScalar.h"
#include "include/NumericalFlux.h"
#include "include/HistoryBinary.h"

// ---------------- Simulation parameters ----------------
static const double a            = 1.0;    // Transport velocity
static const double xL           = -1.0;   // Left interval extrema
static const double xR           =  1.0;   // Right interval extrema
static const int    N            = 999999;  // N+1 = 10^6 finite volumes
static const double CFL          = 1.0;    // max|f'(u)| dt / dx (stable for CFL <= 1). dt is computed from it
static const int    T            = 2001;   // Time steps in the history (final time = dt*(T-1)). Only the recorded rows of the history are filled
static const int    RECORD_EVERY = 100;    // Store and write every RECORD_EVERY time steps (first and last always)
static const char*  OUTPUT_FILE  = "linear_transport.bin";
// -------------------------------------------------------

// Physics, evaluated on n states at once
static void flux(const double* u, double* f, int n)      { for (int j = 0; j < n; ++j) f[j]  = a*u[j]; }
static void fluxGrad(const double* u, double* df, int n) { for (int j = 0; j < n; ++j) df[j] = a; (void)u; }

static double initialCondition(double x) { return (x < 0.0) ? 1.0 : 0.0; }

static double exactSolution(double x, double t)
{
    double L  = xR - xL;
    double xi = x - a*t;
    xi = xi - L*floor((xi - xL)/L); // Wrap back into [xL, xR)
    return initialCondition(xi);
}

int main(void)
{
    double dx = (xR - xL)/(N+1);

    // Initial condition, evaluated at the cell centers
    double* u0 = malloc(sizeof(double)*(N+1));
    double* u  = malloc(sizeof(double)*HISTORY_SIZE(T, N));
    for (int j = 0; j <= N; ++j) u0[j] = initialCondition(xL + (j + 0.5)*dx);

    // Time step from the CFL number. The wave speed is f'(u) = a everywhere
    double dt     = CFL*dx/fabs(a);
    double tFinal = dt*(T-1);

    // Solve and time the whole history computation
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DScalar(N, u0, flux, fluxGrad, RUSANOV, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // Relative L1 error at the final time
    double errSum = 0.0, exSum = 0.0;
    for (int j = 0; j <= N; ++j) {
        double ue = exactSolution(xL + (j + 0.5)*dx, tFinal);
        errSum += fabs(u[IDX(T-1, j, N)] - ue);
        exSum  += fabs(ue);
    }

    printf("dt = %g, final time = %g\n", dt, tFinal);
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Time per cell update: %.3f ns\n", 1e9*elapsed/((double)(T-1)*(N+1)));
    printf("Relative L1 error at t = %g: %.6e\n", tFinal, errSum/exSum);

    writeHistoryBinary(OUTPUT_FILE, u, N, T, xL, xR, dt, RECORD_EVERY);

    free(u0);
    free(u);
    return 0;
}
