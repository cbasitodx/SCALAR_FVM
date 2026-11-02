// Implementation of the numerical fluxes. Not compiled on its own: included at the end of include/NumericalFluxCUDA.cuh

template <class Physics>
__device__ double Rusanov<Physics>::F(double uL, double uR)
{
    return (Physics::flux(uL) + Physics::flux(uR))/2 - max(fabs(Physics::fluxGrad(uL)), fabs(Physics::fluxGrad(uR)))/2 * (uR - uL);
}
