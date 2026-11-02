#ifndef FVM1D_SCALAR_H
#define FVM1D_SCALAR_H

#include "include/Index.h"              // IDX, HISTORY_SIZE, isRecorded (common/)
#include "include/BoundaryConditions.h" // BCType, fillGhosts (common/)
#include "NumericalFlux.h"

// Inline version of the solver: one pass over the cells per time step, with the physics inlined.
// Same arguments and history layout as the batched version (cpu_batch/), except that the physics
// and the numerical flux work on one value at a time (see NumericalFlux.h).
// It works on two rows (the old and the new state, which swap roles every step) and copies only the
// recorded time steps into the history, like the CUDA version.
static inline void FVM1DScalar(
    int N,                      // IN: N+1 is the number of finite volumes, and N is the number of internal points between xL and xR
    const double* restrict u_0, // IN: Initial state: array with N + 1 (from 0 to N) elements. Is the function averaged in the N + 1 volumes
    Flux flux,                  // IN: Flux of the conservation law (one value)
    FluxGrad flux_grad,         // IN: Gradient of the flux (one value)
    NumericalFlux F,            // IN: Numerical flux (e.g. rusanov1DScalar), see NumericalFlux.h
    double xL,                  // IN: Left interval extrema  (xL < xR)
    double xR,                  // IN: Right interval extrema (xL < xR)
    double delta_t,             // IN: Temporal time step (CFL is not verified by the solver automatically)
    int T,                      // IN: Number of time steps of the history, including the initial state: t^0, ..., t^{T-1}
                                //     (initial time is 0, so the final time is Delta t * (T-1))
    BCType bc,                  // IN: Boundary condition type
    int recordEvery,            // IN: Store only the time steps n with isRecorded(n, T, recordEvery) (see Index.h)
    double* restrict u          // OUT: Flat history array of HISTORY_SIZE(T, N) elements (see Index.h), allocated by the caller.
                                //      Must not overlap u_0. u[IDX(n, j, N)] is the approximated function at cell c_j and time t^n.
                                //      Only the cells of the recorded rows are filled (the rest is left untouched)
);

// Implementation (in the header so the compiler sees it where it is called: the function pointers
// passed by the caller are then constants, and the compiler inlines them)
#include "src/FVM1DScalar.inl"

#endif
