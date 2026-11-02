// Implementation of FVM1DScalarCUDA. Not compiled on its own: included at the end of include/FVM1DScalarCUDA.cuh

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Macro. Abort with a message if a CUDA API call fails
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// One time step, fused. For each cell j = 0..N (grid-stride loop), one thread:
//   1. reads uo[j-1], uo[j], uo[j+1]       (the ghost cells uo[-1] and uo[N+1] are already filled by the previous kernel)
//   2. computes FL = NumFlux::F(uo[j-1], uo[j])  and  FR = NumFlux::F(uo[j], uo[j+1])
//   3. writes   un[j] = uo[j] - lambda*(FR - FL)
//   4. calls    fillGhostsFromCell(un, j, N, bc), so the ghost cells of the new row are ready for the next step
// No thread depends on another one: all of them only read the previous row, which the previous launch completed.
template <class NumFlux>
__global__ void stepKernel(const double* __restrict__ uo, double* __restrict__ un, int N, double lambda, BCType bc)
{
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; // Index of the thread within the grid
    int numThreadsInGrid = gridDim.x * blockDim.x;               // Number of threads in the grid
    for(int j = globalThreadIdx; j <= N; j+=numThreadsInGrid) { // Stride loop
        // Compute the flux
        double FL = NumFlux::F(uo[j-1], uo[j]  );
        double FR = NumFlux::F(uo[j]  , uo[j+1]);

        // Update the state
        un[j] = uo[j] - lambda*(FR - FL);

        // If it corresponds, update the ghost cells for the new state un
        fillGhostsFromCell(un, j, N, bc);
    }
}

template <class NumFlux>
void FVM1DScalarCUDA(int N, const double* u_0, double xL, double xR, double dt, int T, BCType bc, int recordEvery,
                     double* u, int threadsPerBlock, int numBlocks, double* kernelSeconds)
{
    double dx       = (xR - xL)/(N+1);
    double lambda   = dt/dx;
    size_t rowSize  = (size_t)(N+3);           // One row: ghost, N+1 cells, ghost
    size_t rowBytes = sizeof(double)*rowSize;

    // Device memory: only two rows, the old state (t^n) and the new state (t^{n+1}).
    // uOld and uNew point to cell 0 of their row, so uOld[-1] and uOld[N+1] are its ghost cells (same for uNew)
    double* d_rows;
    CUDA_CHECK(cudaMalloc((void**)&d_rows, 2*rowBytes));
    double* uOld = d_rows + 1;
    double* uNew = d_rows + rowSize + 1;

    // Row 0 with its ghost cells: built on the host, stored in the host history (it is always recorded)
    // and copied to the device as the first old state
    double* row0 = u + IDX(0, -1, N);
    memcpy(u + IDX(0, 0, N), u_0, sizeof(double)*(size_t)(N+1));
    fillGhosts(u + IDX(0, 0, N), N, bc);
    CUDA_CHECK(cudaMemcpy(uOld - 1, row0, rowBytes, cudaMemcpyHostToDevice));

    // Time loop on the host: one kernel launch per step. A launch starts only when the previous one has finished,
    // which is the only synchronization the scheme needs (each step reads the whole previous row).
    // Kernel time is measured in segments between copies, so the copies are not counted in it
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double kernelMs = 0.0;
    CUDA_CHECK(cudaEventRecord(start));

    for (int t_n = 0; t_n < T - 1; ++t_n) {
        // Compute t^{n+1} from t^n
        stepKernel<NumFlux><<<numBlocks, threadsPerBlock>>>(uOld, uNew, N, lambda, bc);

        if (isRecorded(t_n + 1, T, recordEvery)) { // t_n + 1 because first state is always recorded
            // Close the kernel time segment (this also waits for the kernels and reports their errors)
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            kernelMs += ms;

            // Copy the new row (ghost cells included) to its place in the host history
            CUDA_CHECK(cudaMemcpy(u + IDX(t_n + 1, -1, N), uNew - 1, rowBytes, cudaMemcpyDeviceToHost));

            // Open the next kernel time segment
            CUDA_CHECK(cudaEventRecord(start));
        }

        // Swap the rows: the new state becomes the old one for the next step,
        // and the old row (no longer needed) will be overwritten by the next new state
        double* tmp = uOld;
        uOld = uNew;
        uNew = tmp;
    }

    // Wait for the GPU to finish. It is already idle here (the last step is always recorded, so it was
    // synchronized and copied above); kept as a safeguard in case the recording rule changes
    CUDA_CHECK(cudaDeviceSynchronize());

    if (kernelSeconds != NULL) *kernelSeconds = 1e-3*kernelMs;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rows));
}
