#ifndef NUMERICAL_FLUX_H
#define NUMERICAL_FLUX_H

// All these functions work on whole arrays: the solver calls them once per time step,
// not once per cell or interface (calls are cheap only when they are few).

// Physics: evaluate the flux, or its derivative, at n states
typedef void (*Flux)(
    const double* u, // IN: States u[0..n-1]
    double* f,       // OUT: f[j] = f(u[j])
    int n            // IN: Number of states
);
typedef void (*FluxGrad)(
    const double* u, // IN: States u[0..n-1]
    double* df,      // OUT: df[j] = f'(u[j])
    int n            // IN: Number of states
);

// A numerical flux: a function that computes the N+2 interface fluxes of one time step,
// and the amount of temporary memory ("scratch") that this function needs.
// The solver allocates the scratch once, so each scheme can use as much as it needs
// without the solver knowing what it is used for.
typedef struct {
    void (*compute)(
        const double* U, // IN: State with ghost cells: U[-1], U[0..N] (the cells), U[N+1]
        int N,           // IN: N+1 is the number of finite volumes
        Flux f,          // IN: Flux function
        FluxGrad df,     // IN: Gradient of the flux function
        double* Fi,      // OUT: Fi[i] = F_{i-1/2}, the flux between cells i-1 and i, for i = 0..N+1
        double* scratch  // IN: Temporary memory of scratchPerCell*(N+3) doubles, given by the solver
    );
    int scratchPerCell;  // Doubles of scratch needed per cell of the state with ghost cells (N+3 cells)
} NumericalFlux;

// Rusanov flux: F_{i-1/2} = (f(U[i-1]) + f(U[i]))/2 - max(|f'(U[i-1])|, |f'(U[i])|)/2 * (U[i] - U[i-1])
// Scratch: f and f' of every cell (2 per cell)
extern const NumericalFlux RUSANOV;

#endif
