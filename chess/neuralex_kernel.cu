/*
* ��Լ�㷨 ����ÿ�����
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
* ����A��ÿһ�к�B���
* B��ֻ��һ�еľ���
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
* �����
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
* ���������
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
* ����ƫ�Ƶ���
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
* �����
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
* ���º���
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