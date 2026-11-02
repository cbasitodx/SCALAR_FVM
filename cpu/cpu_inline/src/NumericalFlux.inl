// Implementation of the numerical fluxes. Not compiled on its own: included at the end of include/NumericalFlux.h

#include <math.h>

static inline double rusanov1DScalar(double uL, double uR, Flux f, FluxGrad df)
{
    return (f(uL) + f(uR))/2 - (fmax(fabs(df(uL)), fabs(df(uR)))/2)*(uR - uL);
}
