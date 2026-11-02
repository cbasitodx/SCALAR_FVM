// Test: Burgers equation u_t + (u^2/2)_x = 0, periodic BCs, Rusanov flux (inline solver).
// Initial condition: u = 1 if x < 0, u = 0 if x > 0.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/FVM1DScalar.h"
#include "include/NumericalFlux.h"
#include "include/HistoryBinary.h"

// ---------------- Simulation parameters ----------------
static const double xL           = -1.0;   // Left interval extrema
static const double xR           =  1.0;   // Right interval extrema
static const int    N            = 999999;  // N+1 = 10^6 finite volumes
static const double CFL          = 1.0;    // max|f'(u)| dt / dx (stable for CFL <= 1). dt is computed from it
static const int    T            = 2001;   // Time steps in the history (final time = dt*(T-1)). Only the recorded rows of the history are filled
static const int    RECORD_EVERY = 100;    // Store and write every RECORD_EVERY time steps (first and last always)
static const char*  OUTPUT_FILE  = "burgers.bin";
// -------------------------------------------------------

// Physics, evaluated on one state
static double flux(double u)     { return 0.5*u*u; }
static double fluxGrad(double u) { return u; }

static double initialCondition(double x) { return (x < 0.0) ? 1.0 : 0.0; }

int main(void)
{
    double dx = (xR - xL)/(N+1);

    // Initial condition, evaluated at the cell centers
    double* u0 = malloc(sizeof(double)*(N+1));
    double* u  = malloc(sizeof(double)*HISTORY_SIZE(T, N));
    for (int j = 0; j <= N; ++j) u0[j] = initialCondition(xL + (j + 0.5)*dx);

    // Time step from the CFL number. The scheme keeps u within the initial range,
    // so max|f'(u0)| bounds the wave speed for the whole run
    double maxSpeed = 0.0;
    for (int j = 0; j <= N; ++j) maxSpeed = fmax(maxSpeed, fabs(fluxGrad(u0[j])));
    double dt = CFL*dx/maxSpeed;

    // Solve and time the whole history computation
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    FVM1DScalar(N, u0, flux, fluxGrad, rusanov1DScalar, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u);
    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    printf("dt = %g, final time = %g\n", dt, dt*(T-1));
    printf("Elapsed time: %.3f ms\n", 1e3*elapsed);
    printf("Time per cell update: %.3f ns\n", 1e9*elapsed/((double)(T-1)*(N+1)));

    writeHistoryBinary(OUTPUT_FILE, u, N, T, xL, xR, dt, RECORD_EVERY);

    free(u0);
    free(u);
    return 0;
}
