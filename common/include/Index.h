#ifndef INDEX_H
#define INDEX_H

#include <stddef.h>

// Layout of the flat history array u. Each time step is a row of N+3 values:
//
//     [ ghost | cell 0, cell 1, ..., cell N | ghost ]
//
// The two ghost cells are stored in the row, so the solver can fill them in place (no copy of the row).
//
// Usage: u[IDX(n, j, N)] is the cell average of cell c_j at time t^n, for j = 0..N.
//        j = -1 and j = N+1 are the ghost cells.
#define IDX(n, j, N) ((size_t)(n) * ((size_t)(N) + 3) + (size_t)((j) + 1))

// Number of doubles to allocate for a history of T time steps with N+1 cells (ghost cells included)
#define HISTORY_SIZE(T, N) ((size_t)(T) * ((size_t)(N) + 3))

// True if time step n (of a history of T steps) is recorded when recording every 'every' steps.
// The first and last steps are always recorded. This is the single rule shared by the binary writer
// (which writes only these rows) and the CUDA solver (which copies only these rows back to the host).
static inline int isRecorded(int n, int T, int every) { return (n % every == 0) || (n == T - 1); }

#endif
