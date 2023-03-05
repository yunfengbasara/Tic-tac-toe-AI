#include "cuda_runtime.h"
#include "neuralex.h"

using namespace util;
extern "C" __global__ void matrixSumCuda2(CUDAMatrix a, CUDAMatrix c)
{
    int bx = blockIdx.x;
    int bdx = blockDim.x;
    int tx = threadIdx.x;

    extern __shared__ float st[];
    st[tx] = a.data[a.pitchcols * bx + tx];
    __syncthreads();

    for (int sep = 1; sep < bdx; sep *= 2) {
        if ((tx % (2 * sep)) != 0) {
            break;
        }

        if (tx + sep < bdx) {
            st[tx] += st[tx + sep];
        }
        __syncthreads();
    }

    if (tx == 0) {
        int bIdx = c.pitchcols * bx;
        c.data[bIdx + 0] = st[0];
    }
}