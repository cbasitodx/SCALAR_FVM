#ifndef FVM1D_SCALAR_H
#define FVM1D_SCALAR_H

#include "include/Index.h"              // IDX, HISTORY_SIZE, isRecorded (common/)
#include "include/BoundaryConditions.h" // BCType, fillGhosts (common/)
#include "NumericalFlux.h"

// Batched version of the solver. It works on two rows (the old and the new state, which swap roles every step)
// and copies only the recorded time steps into the history, like the CUDA version.
void FVM1DScalar(
    int N,                      // IN: N+1 is the number of finite volumes, and N is the number of internal points between xL and xR
    const double* restrict u_0, // IN: Initial state: array with N + 1 (from 0 to N) elements. Is the function averaged in the N + 1 volumes
    Flux flux,                  // IN: Flux of the conservation law (evaluates a whole array of states)
    FluxGrad flux_grad,         // IN: Gradient of the flux (evaluates a whole array of states)
    NumericalFlux F,            // IN: Numerical flux (e.g. RUSANOV), see NumericalFlux.h
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

#endif