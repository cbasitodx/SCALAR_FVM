#ifndef HISTORY_BINARY_H
#define HISTORY_BINARY_H

// C linkage, so this C function can also be called from C++/CUDA code
#ifdef __cplusplus
extern "C" {
#endif

// Writes the history array u (as filled by the solvers) to a binary file.
// Only every 'every'-th time step is written (the last time step T-1 is always written). Ghost cells are not written.
//
// File layout (native byte order, little-endian on x86 and Apple Silicon):
//     int64   number of cells (N+1)
//     int64   number of records (time steps written)
//     double  xL, xR            (cell j is centered at xL + (j + 0.5)*(xR - xL)/(N+1))
//     then, for each record:    double t, followed by the N+1 cell values u_0 .. u_N
//
// Returns 0 on success, 1 if the file could not be opened or written.
int writeHistoryBinary(
    const char* filename, // IN: Output file name
    const double* u,      // IN: History array, u[IDX(n, j, N)]
    int N,                // IN: N+1 is the number of finite volumes
    int T,                // IN: Number of time steps in the history (rows 0 to T-1)
    double xL,            // IN: Left interval extrema
    double xR,            // IN: Right interval extrema
    double dt,            // IN: Temporal time step
    int every             // IN: Record every 'every' time steps
);

#ifdef __cplusplus
}
#endif

#endif
