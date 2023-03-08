/*
* 规约算法 矩阵每行求和
* C[x][0] = C[x][0] + C[x][1] + ... + C[x][n]
*/
extern "C" __global__ void reduction(
    float* a, int apw, 
    float* c, int cpw)
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

/*
* 矩阵乘法
*/
#define BLOCK_SIZE 32
extern "C" __global__ void matrixMul(
    float* a, int apw, 
    float* b, int bpw, 
    float* c, int cpw)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = apw * BLOCK_SIZE * by;
    int aStep = BLOCK_SIZE;
    int aEnd = aBegin + apw - 1;

    int bBegin = bx * BLOCK_SIZE;
    int bStep = bpw * BLOCK_SIZE;

    float sumC = 0;

    for (int ia = aBegin, ib = bBegin;
        ia <= aEnd; ia += aStep, ib += bStep) {
        __shared__ float AS[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float BS[BLOCK_SIZE][BLOCK_SIZE];

        AS[ty][tx] = a[ia + ty * apw + tx];
        BS[ty][tx] = b[ib + ty * bpw + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sumC += AS[ty][k] * BS[k][tx];
        }

        __syncthreads();
    }

    int cBlock = cpw * BLOCK_SIZE * by + bx * BLOCK_SIZE;
    int cThread = cpw * ty + tx;
    c[cBlock + cThread] = sumC;
}

/*
* 矩阵A的每一列和B相加
* B是只有一列的矩阵
* C[x][n] = A[x][n] + B[x][0]
*/
extern "C" __global__ void colwiseAdd(
    float* a, int apw,
    float* b, int bpw,
    float* c, int cpw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    __shared__ float tmp;
    if (tx == 0) {
        int bIdx = bpw * bx;
        tmp = b[bIdx];
    }
    __syncthreads();

    c[cpw * bx + tx] = a[apw * bx + tx] + tmp;
}

/*
* 激活函数
*/
extern "C" __global__ void activation(
    float* a, int apw,
    float* b, int bpw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[apw * bx + tx];
    tmp = 1.0f / (1.0f + expf(-tmp));

    b[bpw * bx + tx] = tmp;
}