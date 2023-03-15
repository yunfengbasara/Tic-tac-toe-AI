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
* Eigen�������Կ��ڴ��������ǰ��д洢
* Ȼ���������ǰ��м���,�����������������ȷ
* tmpӦ����colwiseAdd2����һ��,ȡ��
* �������
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
* ����A��ÿһ�к�B���
* B��ֻ��һ�еľ���
* C[x][n] = A[x][n] + B[x][0]
* �ڴ�������
*/
extern "C" __global__ void colwiseAdd2(
    float* a, int aw, int ah,
    float* b, int bw, int bh,
    float* c, int cw, int ch,
    int block_x, int block_y)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if ((bx * block_x + tx) >= aw) {
        return;
    }

    if ((by * block_y + ty) >= ah) {
        return;
    }

    int cBlock = by * block_y * cw + bx * block_x;
    int cIndex = cBlock + ty * cw + tx;
    c[cIndex] = a[cIndex] + b[cIndex % bh];
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
extern "C" __global__ void deltaQuadratic(
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

extern "C" __global__ void deltaCrossEntropy(
    float* a, int aw,
    float* b, int bw,
    float* c, int cw)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx] - b[bw * bx + tx];

    c[cw * bx + tx] = tmp;
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
* ����bias����
*/
extern "C" __global__ void updateBias(
    float* a, int aw,
    float* b, int bw,
    float eta, int batch)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx] * eta / batch;
    b[bw * bx + tx] -= tmp;
}

/*
* ����weight����
*/
extern "C" __global__ void updateWeight(
    float* a, int aw,
    float* b, int bw,
    float eta, int batch,
    float lambda, int items)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float tmp = a[aw * bx + tx] * eta / batch;
    float w = (1 - eta * (lambda / items)) * b[bw * bx + tx];
    b[bw * bx + tx] = w - tmp;
}