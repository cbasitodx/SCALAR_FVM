#ifndef FVM1D_SCALAR_CUDA_CUH
#define FVM1D_SCALAR_CUDA_CUH

#include "include/Index.h"              // IDX, HISTORY_SIZE, isRecorded (common/), same history layout as the CPU versions
#include "include/BoundaryConditions.h" // BCType, fillGhosts, fillGhostsFromCell (common/)
#include "include/NumericalFluxCUDA.cuh"

// CUDA version of FVM1DScalar. Same arguments and history layout as the CPU versions, except:
//   - the numerical flux (which carries the physics) is a template parameter, so it is inlined into the kernel,
//   - only the recorded time steps are copied back to the host (see recordEvery),
//   - the launch configuration is given explicitly,
//   - the time spent in the kernels is returned, measured on the GPU.
//
// One fused kernel per time step: each thread computes whole cells (both interface fluxes and the update).
// The GPU only holds two rows (the old and the new state, which swap roles every step), so its memory
// does not limit T; each recorded row is copied to its place in the host history as soon as it is computed.
//
// Usage: FVM1DScalarCUDA< Rusanov<Burgers> >(N, u_0, xL, xR, dt, T, BC_PERIODIC, RECORD_EVERY, u,
//                                             threadsPerBlock, numBlocks, &kernelSeconds);
template <class NumFlux>
void FVM1DScalarCUDA(
    int N,                 // IN: N+1 is the number of finite volumes
    const double* u_0,     // IN: Initial state, N+1 elements (host array)
    double xL,             // IN: Left interval extrema  (xL < xR)
    double xR,             // IN: Right interval extrema (xL < xR)
    double dt,             // IN: Temporal time step (CFL is not verified by the solver automatically)
    int T,                 // IN: Number of time steps of the history, including the initial state
    BCType bc,             // IN: Boundary condition type
    int recordEvery,       // IN: Copy back only the time steps n with isRecorded(n, T, recordEvery) (see Index.h)
    double* u,             // OUT: History array of HISTORY_SIZE(T, N) elements (host array), u[IDX(n, j, N)].
                           //      Only the recorded rows are filled (the others are left untouched, and cost no
                           //      physical memory as long as nobody writes them)
    int threadsPerBlock,   // IN: Threads per block (multiple of 32)
    int numBlocks,         // IN: Number of blocks (any value works: the kernel uses a grid-stride loop)
    double* kernelSeconds  // OUT: Time spent in the kernels, measured on the GPU (copies excluded). May be NULL
);

// Implementation (in the header: template code must be visible where it is used)
#include "src/FVM1DScalarCUDA.inl"

#endif
