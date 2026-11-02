#ifndef NUMERICAL_FLUX_H
#define NUMERICAL_FLUX_H

// Inline version: the physics and the numerical flux work on one value at a time.
// Their code is in the headers (src/*.inl), so the compiler sees it where the solver uses it
// and inlines it into the solver loop (the function pointers become constants at the call site).

typedef double (*Flux)(double);     // f(u)
typedef double (*FluxGrad)(double); // f'(u)
typedef double (*NumericalFlux)(
    double uL,  // State left of the interface
    double uR,  // State right of the interface
    Flux f,     // Flux function
    FluxGrad df // Gradient of the flux function
);

// Rusanov flux: (f(uL) + f(uR))/2 - max(|f'(uL)|, |f'(uR)|)/2 * (uR - uL)
static inline double rusanov1DScalar(double uL, double uR, Flux f, FluxGrad df);

// Implementation (in the header so the compiler can inline it where it is used)
#include "src/NumericalFlux.inl"

#endif
