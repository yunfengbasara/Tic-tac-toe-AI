#include "neuralex.h"
#include "util.h"
#include <Windows.h>
#include <chrono>
#include <iostream>
#include "cuda_runtime.h"

using namespace Eigen;
using namespace util;
using namespace std;
using namespace std::chrono;

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char* file, const int line) {
    if (CUDA_SUCCESS != err) {
        const char* errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}

#endif

util::NeuralEx::NeuralEx()
{
    std::vector<char> cubin;
    if (!CompileFileToCUBIN(L"neuralex_kernel.cu", cubin)) {
        return;
    }

    if (!LoadCUBIN(cubin, m_nModule)) {
        return;
    }

    checkCudaErrors(cuModuleGetFunction(
        &m_fMatrixMul, m_nModule, "matrixMul"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fReduction, m_nModule, "reduction"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fColwiseAdd, m_nModule, "colwiseAdd"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fActivation, m_nModule, "activation"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fActivatePrime, m_nModule, "activatePrime"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fDeltaTarget, m_nModule, "deltaTarget"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fMulTransB, m_nModule, "mulTransB"));
    
}

util::NeuralEx::~NeuralEx()
{
    Release();
}

bool util::NeuralEx::InitBuild(std::vector<int> p)
{
    if (p.size() < 3) {
        return false;
    }

    Release();

    cuStreamCreate(&m_nStream, cudaStreamNonBlocking);

    m_vNetParam = p;

    // 构建网络
    for (size_t i = 1; i < p.size(); i++) {
        HOSTMatrix hm = HOSTMatrix::Random(p[i], 1);
        CUDAMatrix cm = CreateCUDAMatrix(hm);
        m_vBiases.push_back(cm);
    }

    for (size_t i = 1; i < p.size(); i++) {
        HOSTMatrix hm = HOSTMatrix::Zero(p[i], 1);
        CUDAMatrix cm = CreateCUDAMatrix(hm);
        m_vNabla_b.push_back(cm);
    }

    for (size_t i = 1; i < p.size(); i++) {
        HOSTMatrix hm = HOSTMatrix::Random(p[i], p[i - 1]);
        CUDAMatrix cm = CreateCUDAMatrix(hm);
        m_vWeights.push_back(cm);
    }

    for (size_t i = 1; i < p.size(); i++) {
        HOSTMatrix hm = HOSTMatrix::Zero(p[i], p[i - 1]);
        CUDAMatrix cm = CreateCUDAMatrix(hm);
        m_vNabla_w.push_back(cm);
    }

    // 构建输入输出
    // 由于batch可变,默认先建立batch=1
    m_nBatch = 128;

    for (size_t i = 0; i < p.size(); i++) {
        //CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        HOSTMatrix hm = HOSTMatrix::Random(p[i], m_nBatch);
        CUDAMatrix cm = CreateCUDAMatrix(hm);
        m_vActivations.push_back(cm);
    }

    for (size_t i = 0; i < p.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        m_vDelta.push_back(cm);
    }

    CUDAMatrix z = m_vActivations.back();
    m_nSG = CreateCUDAMatrix(z.height, z.width);

    for (size_t i = 1; i < p.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        m_vInputSum.push_back(cm);
    }

    m_mTarget = CreateCUDAMatrix(p.back(), m_nBatch);
    
    return true;
}

bool util::NeuralEx::SetSample(HOSTMatrix& in, HOSTMatrix& target)
{
    if (in.cols() != target.cols()) {
        return false;
    }

    int batch = in.cols();

    // 检查输入维度
    if (m_vActivations[0].height != in.rows()) {
        return false;
    }

    // 检查输出维度
    if (m_mTarget.height != target.rows()) {
        return false;
    }

    // 批相同
    if (m_nBatch == batch) {
        CopyHostToCUDA(in, m_vActivations[0]);
        CopyHostToCUDA(target, m_mTarget);
        return true;
    }

    // 批不同 重新构建激活层和总和层
    m_nBatch = batch;

    // 构建激活层
    for (auto& item : m_vActivations) {
        DestroyCUDAMatrix(item);
    }
    m_vActivations.clear();

    for (auto& item : m_vDelta) {
        DestroyCUDAMatrix(item);
    }
    m_vDelta.clear();

    DestroyCUDAMatrix(m_nSG);

    for (size_t i = 0; i < m_vNetParam.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(m_vNetParam[i], m_nBatch);
        m_vActivations.push_back(cm);
    }

    for (size_t i = 0; i < m_vNetParam.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(m_vNetParam[i], m_nBatch);
        m_vDelta.push_back(cm);
    }

    CUDAMatrix z = m_vActivations.back();
    m_nSG = CreateCUDAMatrix(z.height, z.width);

    // 构建输入和层
    for (auto& item : m_vInputSum) {
        DestroyCUDAMatrix(item);
    }
    m_vInputSum.clear();

    for (size_t i = 1; i < m_vNetParam.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(m_vNetParam[i], m_nBatch);
        m_vInputSum.push_back(cm);
    }

    CopyHostToCUDA(in, m_vActivations[0]);
    CopyHostToCUDA(target, m_mTarget);
    return true;
}

void util::NeuralEx::SetLearnRate(float eta)
{
    m_fEta = eta;
}

void util::NeuralEx::SGD()
{
    // 前向传播
    FeedForward();

    // 反向矫正
    BackPropLast();
    BackPropLine();

    // batch处理完后更新参数
    Update();
}

void util::NeuralEx::FeedForward()
{
    // m_vActivations第一层为输入层
    // 每层使用S型激活函数
    // S型激活函数:f(x) = 1/(1+e^(-x))
    // S型激活函数导数:f(x)' = f(x)*(1 - f(x))
    for (size_t i = 0; i < m_vNetParam.size() - 1; i++) {
        int width = m_vInputSum[i].width;
        int height = m_vInputSum[i].height;

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        int gx = (width + block.x - 1) / block.x;
        int gy = (height + block.y - 1) / block.y;

        // 输出和权重乘积
        void* mulparams[] = 
        {
            (void*)&m_vWeights[i].data, (void*)&m_vWeights[i].pitchwidth,
            (void*)&m_vActivations[i].data, (void*)&m_vActivations[i].pitchwidth,
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].pitchwidth
        };

        checkCudaErrors(cuLaunchKernel(m_fMatrixMul,
            gx, gy, 1, block.x, block.y, 1,
            0, m_nStream, &mulparams[0], 0));
        checkCudaErrors(cuStreamSynchronize(m_nStream));

        // 计算偏移
        void* addparams[] = 
        {
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].pitchwidth,
            (void*)&m_vBiases[i].data, (void*)&m_vBiases[i].pitchwidth,
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].pitchwidth
        };

        checkCudaErrors(cuLaunchKernel(m_fColwiseAdd,
            height, 1, 1, width, 1, 1,
            0, m_nStream, &addparams[0], 0));
        checkCudaErrors(cuStreamSynchronize(m_nStream));

        // 激活
        void* actparams[] =
        {
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].pitchwidth,
            (void*)&m_vActivations[i + 1].data, (void*)&m_vActivations[i + 1].pitchwidth
        };

        checkCudaErrors(cuLaunchKernel(m_fActivation,
            height, 1, 1, width, 1, 1,
            0, m_nStream, &actparams[0], 0));
        checkCudaErrors(cuStreamSynchronize(m_nStream));
    }
}

void util::NeuralEx::BackPropLast()
{
    // 求目标误差
    // 误差函数:loss = 1/2*(t - E)^2
    // 误差函数导数:loss' = E - t
    CUDAMatrix z = m_vActivations.back();
    CUDAMatrix t = m_nSG;

    void* params[] =
    {
        (void*)&z.data, (void*)&z.pitchwidth,
        (void*)&t.data, (void*)&t.pitchwidth
    };

    checkCudaErrors(cuLaunchKernel(m_fActivatePrime,
        z.height, 1, 1, z.width, 1, 1,
        0, m_nStream, &params[0], 0));
    checkCudaErrors(cuStreamSynchronize(m_nStream));

    CUDAMatrix d = m_vDelta.back();
    void* deltaparams[] =
    {
        (void*)&z.data, (void*)&z.pitchwidth,
        (void*)&m_mTarget.data, (void*)&m_mTarget.pitchwidth,
        (void*)&t.data, (void*)&t.pitchwidth,
        (void*)&d.data, (void*)&d.pitchwidth,
    };

    checkCudaErrors(cuLaunchKernel(m_fDeltaTarget,
        z.height, 1, 1, z.width, 1, 1,
        0, m_nStream, &deltaparams[0], 0));
    checkCudaErrors(cuStreamSynchronize(m_nStream));

    // 偏移按行求和
    CUDAMatrix b = m_vNabla_b.back();
    void* rowwiseparams[] = 
    {
        (void*)&d.data, (void*)&d.pitchwidth,
        (void*)&b.data, (void*)&b.pitchwidth
    };

    checkCudaErrors(cuLaunchKernel(m_fReduction,
        d.height, 1, 1, d.width, 1, 1,
        d.stride, m_nStream, &rowwiseparams[0], 0));
    checkCudaErrors(cuStreamSynchronize(m_nStream));

    // 求权值更新
    CUDAMatrix lz = *(m_vActivations.rbegin() + 1);
    CUDAMatrix lw = m_vNabla_w.back();

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int gx = (lw.width + block.x - 1) / block.x;
    int gy = (lw.height + block.y - 1) / block.y;

    void* mulparams[] =
    {
        (void*)&d.data, (void*)&d.pitchwidth,
        (void*)&lz.data, (void*)&lz.pitchwidth,
        (void*)&lw.data, (void*)&lw.pitchwidth
    };

    checkCudaErrors(cuLaunchKernel(m_fMulTransB,
        gx, gy, 1, block.x, block.y, 1,
        0, m_nStream, &mulparams[0], 0));
    checkCudaErrors(cuStreamSynchronize(m_nStream));

    //HOSTMatrix ha(lz.height, lz.width);
    //CopyCUDAToHost(lz, ha, m_nStream);
    //HOSTMatrix hb(d.height, d.width);
    //CopyCUDAToHost(d, hb, m_nStream);
    //HOSTMatrix mc1 = hb * ha.transpose();

    //HOSTMatrix mc2(lw.height, lw.width);
    //CopyCUDAToHost(lw, mc2, m_nStream);

    //    float eps = 1.e-4;
    //    const auto& c1 = mc1.array();
    //    const auto& c2 = mc2.array();
    //    for (int i = 0; i < mc1.size(); i++) {
    //        if (fabs(c1(i) - c2(i)) > eps) {
    //            cout << i << endl;
    //            cout << c1(i) << endl;
    //            cout << c2(i) << endl;
    //            break;
    //        }
    //    }
}

void util::NeuralEx::BackPropLine()
{

}

void util::NeuralEx::Update()
{
}

void util::NeuralEx::Release()
{
    for (auto& item : m_vBiases) {
        DestroyCUDAMatrix(item);
    }
    m_vBiases.clear();

    for (auto& item : m_vNabla_b) {
        DestroyCUDAMatrix(item);
    }
    m_vNabla_b.clear();

    for (auto& item : m_vWeights) {
        DestroyCUDAMatrix(item);
    }
    m_vWeights.clear();

    for (auto& item : m_vNabla_w) {
        DestroyCUDAMatrix(item);
    }
    m_vNabla_w.clear();

    for (auto& item : m_vActivations) {
        DestroyCUDAMatrix(item);
    }
    m_vActivations.clear();

    for (auto& item : m_vDelta) {
        DestroyCUDAMatrix(item);
    }
    m_vDelta.clear();
    
    DestroyCUDAMatrix(m_nSG);

    for (auto& item : m_vInputSum) {
        DestroyCUDAMatrix(item);
    }
    m_vInputSum.clear();

    DestroyCUDAMatrix(m_mTarget);

    if (m_nStream != nullptr) {
        cuStreamDestroy(m_nStream);
        m_nStream = nullptr;
    }
}
