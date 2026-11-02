#include <math.h>
#include "include/NumericalFlux.h"

static void rusanov1DScalar(const double* U, int N, Flux f, FluxGrad df, double* Fi, double* scratch)
{
    // f and f' of every cell, ghost cells included (N+3 values each), stored in the scratch.
    // Index shift: fU[k] and dfU[k] belong to cell k-1, so fU[0] is the left ghost cell
    double* fU  = scratch;
    double* dfU = scratch + (N+3);
    f (U - 1, fU,  N+3);
    df(U - 1, dfU, N+3);

    // Interface i-1/2 is between cells i-1 (index i in fU, dfU) and i (index i+1)
    for (int i = 0; i <= N+1; ++i) {
        Fi[i] = (fU[i] + fU[i+1])/2 - (fmax(fabs(dfU[i]), fabs(dfU[i+1]))/2)*(U[i] - U[i-1]);
    }
}

const NumericalFlux RUSANOV = { rusanov1DScalar, 2 };
