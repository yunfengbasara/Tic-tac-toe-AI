/*
* 规约算法 矩阵每行求和
* C[x][0] = C[x][0] + C[x][1] + ... + C[x][n]
*/
extern "C" __global__ void reduction(float* a, int apw, float* c, int cpw)
{
    int bx = blockIdx.x;
    int bdx = blockDim.x;
    int tx = threadIdx.x;

    extern __shared__ float st[];
    st[tx] = a[apw * bx + tx];
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
        int bIdx = cpw * bx;
        c[bIdx + 0] = st[0];
    }
}