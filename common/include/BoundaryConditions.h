#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

// Boundary conditions supported by the solvers (shared by cpu/ and cuda/).
// The ghost cells U[-1] and U[N+1] of a row are stored next to it (see Index.h).

typedef enum { BC_PERIODIC } BCType;

// Marks a function as usable both on the CPU and inside GPU kernels when compiled by nvcc.
// In plain C (cpu/) it expands to nothing.
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// Fills both ghost cells of the row U (U points to cell 0), once all its cells are known.
// Used by the CPU solvers at every step, and by the CUDA solver for the initial row.
static inline HOST_DEVICE void fillGhosts(double* U, int N, BCType bc)
{
    switch (bc) {
    case BC_PERIODIC: U[-1] = U[N]; U[N+1] = U[0]; break;
    }
}

// Writes the ghost cells of the row U that depend on cell j, right after cell j has been computed.
// Lets each GPU thread fill the ghosts of the row it is writing, without waiting for the other threads.
static inline HOST_DEVICE void fillGhostsFromCell(double* U, int j, int N, BCType bc)
{
    switch (bc) {
    case BC_PERIODIC:
        if (j == N) U[-1]  = U[N]; // Left ghost is a copy of the last cell
        if (j == 0) U[N+1] = U[0]; // Right ghost is a copy of the first cell
        break;
    }
}

#endif
