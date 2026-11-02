#ifndef NUMERICAL_FLUX_CUDA_CUH
#define NUMERICAL_FLUX_CUDA_CUH

// Numerical fluxes for the CUDA solver.
//
// A numerical flux is a type with a device function 'static double F(double uL, double uR)'
// that the solver calls at every interface. It is built on top of a physics type, which provides
// 'static double flux(double u)' and 'static double fluxGrad(double u)'
// (see the tests). Both are template parameters, so they are fixed at compile time and the compiler inlines them.
// This is the compile-time counterpart of the CPU version, where the same functions are passed as pointers (for batch)
// or inlined (for inline).

// Rusanov flux for the physics P:  F(uL, uR) = (f(uL) + f(uR))/2 - max(|f'(uL)|, |f'(uR)|)/2 * (uR - uL)
template <class Physics>
struct Rusanov {
    __device__ static double F(double uL, double uR);
};

// Implementation (in the header: template code must be visible where it is used)
#include "src/NumericalFluxCUDA.inl"

#endif
