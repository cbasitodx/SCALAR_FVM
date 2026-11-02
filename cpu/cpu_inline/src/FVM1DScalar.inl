// Implementation of FVM1DScalar (inline version). Not compiled on its own: included at the end of include/FVM1DScalar.h,
// so the solver is visible where it is called and the compiler can inline the flux functions passed to it.

#include <string.h>
#include <stdlib.h>

static inline void FVM1DScalar(int N, const double* restrict u_0, Flux f, FluxGrad df, NumericalFlux F, double xL, double xR, double dt, int T, BCType bc, int recordEvery, double* restrict u)
{
    // Start by computing the spatial step size
    double dx = (xR - xL)/(N+1);

    // Pre compute the quotient of delta t over delta x
    double lambda = dt / dx;

    // Two rows (ghost cells included): the old state uo (t^n) and the new state un (t^{n+1}).
    // uo and un point to cell 0 of their row, so uo[-1] and uo[N+1] are its ghost cells (same for un)
    size_t  rowSize = (size_t)(N+3);
    double* rows    = malloc(sizeof(double)*2*rowSize);
    double* uo      = rows + 1;
    double* un      = rows + rowSize + 1;

    // Initial state: into the old row, and into the history (row 0 is always recorded)
    memcpy(uo, u_0, sizeof(double)*(N+1));
    memcpy(u + IDX(0, 0, N), u_0, sizeof(double)*(N+1));

    // Main loop
    int c_j; // Variable for iterating in space
    int t_n; // Variable for iterating in time
    for (t_n = 0; t_n < T - 1; ++t_n) { // Row 0 is the initial state, so T rows need T-1 steps
        // Ghost cells of the current row, filled in place
        fillGhosts(uo, N, bc);

        // Single pass over the cells: each interface flux is computed once, and carried to the next cell
        double FL = F(uo[-1], uo[0], f, df); // F_{-1/2}
        for (c_j = 0; c_j <= N; ++c_j) {
            double FR = F(uo[c_j], uo[c_j+1], f, df); // F_{j+1/2}
            un[c_j] = uo[c_j] - lambda*(FR - FL);
            FL = FR; // The right flux of cell j is the left flux of cell j+1
        }

        // Copy the new state to the history if this time step is recorded
        if (isRecorded(t_n + 1, T, recordEvery)) {
            memcpy(u + IDX(t_n + 1, 0, N), un, sizeof(double)*(N+1));
        }

        // Swap the rows: the new state becomes the old one for the next step,
        // and the old row (no longer needed) will be overwritten by the next new state
        double* tmp = uo;
        uo = un;
        un = tmp;
    }

    free(rows);
}
