/*
* 规约算法 矩阵每行求和
* C[x][0] = C[x][0] + C[x][1] + ... + C[x][n]
*/
extern "C" __global__ void reduction(
    float* a, int aw, 
    float* c, int cw)
{
    int bx = blockIdx.x;
    int bdx = blockDim.x;
    int tx = threadIdx.x;

    extern __shared__ float st[];
    st[tx] = a[aw * bx + tx];
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
        int bIdx = cw * bx;
        c[bIdx + 0] = st[0];
    }
}

/*
* 矩阵A的每一列和B相加
* B是只有一列的矩阵
* C[x][n] = A[x][n] + B[x][0]
*/
extern "C" __global__ void colwiseAdd(
    float* a, int aw,
    float* b, int bw,
    float* c, int cw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    __shared__ float tmp;
    if (tx == 0) {
        int bIdx = bw * bx;
        tmp = b[bIdx];
    }
    __syncthreads();

    c[cw * bx + tx] = a[aw * bx + tx] + tmp;
}

/*
* 激活函数
*/
extern "C" __global__ void activation(
    float* a, int aw,
    float* b, int bw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx];
    tmp = 1.0f / (1.0f + expf(-tmp));

    b[bw * bx + tx] = tmp;
}

/*
* 激活函数导数
*/
extern "C" __global__ void activatePrime(
    float* a, int aw,
    float* b, int bw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx];
    tmp = tmp * (1 - tmp);

    b[bw * bx + tx] = tmp;
}

/*
* 计算偏移导数
*/
extern "C" __global__ void deltaTarget(
    float* a, int aw,
    float* b, int bw,
    float* c, int cw,
    float* d, int dw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx] - b[bw * bx + tx];
    tmp *= c[cw * bx + tx];

    d[dw * bx + tx] = tmp;
}

/*
* 逐个乘
*/
extern "C" __global__ void arrayMul(
    float* a, int aw,
    float* b, int bw,
    float* c, int cw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    c[cw * bx + tx] = a[aw * bx + tx] * b[bw * bx + tx];
}

/*
* 更新函数
*/
extern "C" __global__ void update(
    float* a, int aw,
    float* b, int bw,
    float eta, int batch)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx];
    tmp /= batch;
    tmp *= eta;

    b[bw * bx + tx] -= tmp;
}