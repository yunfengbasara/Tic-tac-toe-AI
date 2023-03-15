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
        &m_fColwiseAdd, m_nModule, "colwiseAdd2"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fActivation, m_nModule, "activation"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fActivatePrime, m_nModule, "activatePrime"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fDeltaQuadratic, m_nModule, "deltaQuadratic"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fDeltaCrossEntropy, m_nModule, "deltaCrossEntropy"));

    checkCudaErrors(cuModuleGetFunction(
        &m_fArrayMul, m_nModule, "arrayMul"));
    
    checkCudaErrors(cuModuleGetFunction(
        &m_fUpdate, m_nModule, "update"));
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

    checkCuBlasErrors(cublasCreate(&m_hBlasHandle));

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
    m_nBatch = 1;

    for (size_t i = 0; i < p.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        m_vActivations.push_back(cm);
    }

    for (size_t i = 0; i < p.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        m_vDelta.push_back(cm);
    }

    HOSTMatrix deltaSum = HOSTMatrix(m_nBatch, 1);
    deltaSum.setConstant(1);
    m_nDeltaSum = CreateCUDAMatrix(deltaSum);

    CUDAMatrix z = m_vActivations.back();
    m_nSG = CreateCUDAMatrix(z.rows, z.cols);

    for (size_t i = 1; i < p.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        m_vInputSum.push_back(cm);
    }

    m_mTarget = CreateCUDAMatrix(p.back(), m_nBatch);
    
    return true;
}

bool util::NeuralEx::SetSample(
    HOSTMatrix& in,
    HOSTMatrix& target)
{
    if (in.cols() != target.cols()) {
        return false;
    }

    int batch = in.cols();

    // 检查输入维度
    if (m_vActivations[0].rows != in.rows()) {
        return false;
    }

    // 检查输出维度
    if (m_mTarget.rows != target.rows()) {
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

    DestroyCUDAMatrix(m_nDeltaSum);
    DestroyCUDAMatrix(m_nSG);

    for (size_t i = 0; i < m_vNetParam.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(m_vNetParam[i], m_nBatch);
        m_vActivations.push_back(cm);
    }

    for (size_t i = 0; i < m_vNetParam.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(m_vNetParam[i], m_nBatch);
        m_vDelta.push_back(cm);
    }

    HOSTMatrix deltaSum = HOSTMatrix(m_nBatch, 1);
    deltaSum.setConstant(1);
    m_nDeltaSum = CreateCUDAMatrix(deltaSum);

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

    DestroyCUDAMatrix(m_mTarget);
    m_mTarget = CreateCUDAMatrix(target);
    return true;
}

bool util::NeuralEx::CompareSample(
    HOSTMatrix& in,
    HOSTMatrix& target,
    HOSTMatrix& out,
    float& loss)
{
    if (!SetSample(in, target)) {
        return false;
    }

    FeedForward();

    out = CreateHOSTMatrix(m_vActivations.back());

    switch (m_nCost) {
    // 损失函数:loss = 1/2*(t - E)^2
    case Quadratic: {
        HOSTMatrix E = 1.0f / 2.0f * (target.array() - out.array()).pow(2);
        loss = E.sum() / out.cols();
    }
        break;
    // 损失函数:loss = -ylna - (1-y)ln(1 - a)
    case CrossEntropy: {
        HOSTMatrix E = -target.array() * out.array().log() - (1 - target.array()) * (1 - out.array()).array().log();
        loss = E.sum() / out.cols();
    }
        break;
    }

    return true;
}

void util::NeuralEx::SetLearnRate(float eta)
{
    m_fEta = eta;
}

void util::NeuralEx::SetCostType(CostType type)
{
    m_nCost = type;
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

bool util::NeuralEx::Save(const std::wstring& path)
{
    // 计算保存文件大小
    DWORD filesize = 0;

    // 层数解构:层数+每层节点个数
    filesize += m_vNetParam.size();
    filesize += m_vNetParam.size() * sizeof(DWORD);
    // 学习速度
    filesize += sizeof(m_fEta);
    // Biases大小
    for (auto& b : m_vBiases) {
        filesize += b.size;
    }
    // Weights大小
    for (auto& w : m_vWeights) {
        filesize += w.size;
    }

    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
    LPBYTE lpMem = NULL;

    defer(
        if (lpMem != NULL) {
            ::UnmapViewOfFile(lpMem);
        }
        if (hMap != NULL) {
            ::CloseHandle(hMap);
        }
        if (hFile != INVALID_HANDLE_VALUE) {
            ::CloseHandle(hFile);
        }
    );

    hFile = ::CreateFile(path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    hMap = ::CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, filesize, NULL);
    if (hMap == NULL) {
        return false;
    }

    lpMem = (LPBYTE)::MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (lpMem == NULL) {
        return false;
    }

    // 文件操作位置
    DWORD startfile = 0;

    DWORD layered = m_vNetParam.size();
    *(DWORD*)(lpMem + startfile) = layered;
    startfile += sizeof(layered);

    for (auto& n : m_vNetParam) {
        *(DWORD*)(lpMem + startfile) = n;
        startfile += sizeof(DWORD);
    }

    *(float*)(lpMem + startfile) = m_fEta;
    startfile += sizeof(m_fEta);

    for (auto& b : m_vBiases) {
        HOSTMatrix tb = CreateHOSTMatrix(b);
        memcpy(lpMem + startfile, tb.data(), b.size);
        startfile += b.size;
    }

    for (auto& w : m_vWeights) {
        HOSTMatrix tw = CreateHOSTMatrix(w);
        memcpy(lpMem + startfile, tw.data(), w.size);
        startfile += w.size;
    }

    return true;
}

bool util::NeuralEx::Load(const std::wstring& path)
{
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
    LPBYTE lpMem = NULL;

    defer(
        if (lpMem != NULL) {
            ::UnmapViewOfFile(lpMem);
        }
        if (hMap != NULL) {
            ::CloseHandle(hMap);
        }
        if (hFile != INVALID_HANDLE_VALUE) {
            ::CloseHandle(hFile);
        }
    );

    hFile = ::CreateFile(path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    hMap = ::CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (hMap == NULL) {
        return false;
    }

    lpMem = (LPBYTE)::MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (lpMem == NULL) {
        return false;
    }

    // 文件读取位置
    DWORD startfile = 0;

    // 网络层数
    DWORD layered = *(DWORD*)(lpMem + startfile);
    startfile += sizeof(layered);

    m_vNetParam.resize(layered);
    for (auto& n : m_vNetParam) {
        n = *(DWORD*)(lpMem + startfile);
        startfile += sizeof(DWORD);
    }

    m_fEta = *(float*)(lpMem + startfile);
    startfile += sizeof(m_fEta);

    if (!InitBuild(m_vNetParam)) {
        return false;
    }

    for (auto& b : m_vBiases) {
        float* pf = (float*)(lpMem + startfile);
        HOSTMatrix tb(b.rows, b.cols);
        tb = Map<HOSTMatrix>(pf, b.rows, b.cols);
        CopyHostToCUDA(tb, b);
        startfile += b.size;
    }

    for (auto& w : m_vWeights) {
        float* pf = (float*)(lpMem + startfile);
        HOSTMatrix tw(w.rows, w.cols);
        tw = Map<HOSTMatrix>(pf, w.rows, w.cols);
        CopyHostToCUDA(tw, w);
        startfile += w.size;
    }

    return true;
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

        // 输出和权重乘积
        const float alpha = 1.0f, beta = 0.0f;
        checkCuBlasErrors(cublasSgemm(m_hBlasHandle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_vInputSum[i].rows, m_vInputSum[i].cols, m_vWeights[i].cols, 
            &alpha,
            (const float*)m_vWeights[i].data, m_vWeights[i].rows,
            (const float*)m_vActivations[i].data, m_vActivations[i].rows,
            &beta, 
            (float*)m_vInputSum[i].data, m_vInputSum[i].rows));

        dim3 block(32, 32);
        int gx = (width + block.x - 1) / block.x;
        int gy = (height + block.y - 1) / block.y;

        // 计算偏移
        void* addparams[] = 
        {
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].width,(void*)&m_vInputSum[i].height,
            (void*)&m_vBiases[i].data, (void*)&m_vBiases[i].width,(void*)&m_vBiases[i].height,
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].width,(void*)&m_vInputSum[i].height,
            (void*)&block.x, (void*)&block.y,
        };

        checkCudaErrors(cuLaunchKernel(m_fColwiseAdd,
            gx, gy, 1, block.x, block.y, 1,
            0, nullptr, &addparams[0], 0));

        // 激活
        void* actparams[] =
        {
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].width,
            (void*)&m_vActivations[i + 1].data, (void*)&m_vActivations[i + 1].width
        };

        checkCudaErrors(cuLaunchKernel(m_fActivation,
            height, 1, 1, width, 1, 1,
            0, nullptr, &actparams[0], 0));
    }
}

void util::NeuralEx::BackPropLast()
{
    // 求目标误差
    CUDAMatrix z = m_vActivations.back();
    CUDAMatrix d = m_vDelta.back();

    switch (m_nCost) {
    case Quadratic: {
        void* params[] =
        {
            (void*)&z.data, (void*)&z.width,
            (void*)&m_nSG.data, (void*)&m_nSG.width
        };

        checkCudaErrors(cuLaunchKernel(m_fActivatePrime,
            z.height, 1, 1, z.width, 1, 1,
            0, nullptr, &params[0], 0));

        void* deltaparams[] =
        {
            (void*)&z.data, (void*)&z.width,
            (void*)&m_mTarget.data, (void*)&m_mTarget.width,
            (void*)&m_nSG.data, (void*)&m_nSG.width,
            (void*)&d.data, (void*)&d.width,
        };

        checkCudaErrors(cuLaunchKernel(m_fDeltaQuadratic,
            z.height, 1, 1, z.width, 1, 1,
            0, nullptr, &deltaparams[0], 0));
    }
    break;
    case CrossEntropy: {
        void* deltaparams[] =
        {
            (void*)&z.data, (void*)&z.width,
            (void*)&m_mTarget.data, (void*)&m_mTarget.width,
            (void*)&d.data, (void*)&d.width,
        };

        checkCudaErrors(cuLaunchKernel(m_fDeltaCrossEntropy,
            z.height, 1, 1, z.width, 1, 1,
            0, nullptr, &deltaparams[0], 0));
    }
    break;
    }

    // 偏移按行求和
    CUDAMatrix b = m_vNabla_b.back();
    CUDAMatrix td = m_nDeltaSum;

    // 输出和权重乘积
    const float alpha = 1.0f, beta = 0.0f;
    checkCuBlasErrors(cublasSgemm(m_hBlasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        b.rows, b.cols, d.cols,
        &alpha,
        (const float*)d.data, d.rows,
        (const float*)td.data, td.rows,
        &beta,
        (float*)b.data, b.rows));

    // 求权值更新
    CUDAMatrix lz = *(m_vActivations.rbegin() + 1);
    CUDAMatrix lw = m_vNabla_w.back();
    checkCuBlasErrors(cublasSgemm(m_hBlasHandle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        lw.rows, lw.cols, d.cols,
        &alpha,
        (const float*)d.data, d.rows,
        (const float*)lz.data, lz.rows,
        &beta,
        (float*)lw.data, lw.rows));
}

void util::NeuralEx::BackPropLine()
{
    const float alpha = 1.0f, beta = 0.0f;

    for (size_t i = 1; i < m_vNetParam.size() - 1; i++) {
        CUDAMatrix z = *(m_vActivations.rbegin() + i);
        void* params[] =
        {
            (void*)&z.data, (void*)&z.width,
            (void*)&z.data, (void*)&z.width
        };

        checkCudaErrors(cuLaunchKernel(m_fActivatePrime,
            z.height, 1, 1, z.width, 1, 1,
            0, nullptr, &params[0], 0));

        CUDAMatrix lw = *(m_vWeights.rbegin() + i - 1);
        CUDAMatrix ld = *(m_vDelta.rbegin() + i - 1);
        CUDAMatrix d = *(m_vDelta.rbegin() + i);

        checkCuBlasErrors(cublasSgemm(m_hBlasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d.rows, d.cols, lw.rows,
            &alpha,
            (const float*)lw.data, lw.rows,
            (const float*)ld.data, ld.rows,
            &beta,
            (float*)d.data, d.rows));

        void* arrmularams[] =
        {
            (void*)&d.data, (void*)&d.width,
            (void*)&z.data, (void*)&z.width,
            (void*)&d.data, (void*)&d.width,
        };

        checkCudaErrors(cuLaunchKernel(m_fArrayMul,
            d.height, 1, 1, d.width, 1, 1,
            0, nullptr, &arrmularams[0], 0));

        CUDAMatrix b = *(m_vNabla_b.rbegin() + i);
        CUDAMatrix td = m_nDeltaSum;

        checkCuBlasErrors(cublasSgemm(m_hBlasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            b.rows, b.cols, d.cols,
            &alpha,
            (const float*)d.data, d.rows,
            (const float*)td.data, td.rows,
            &beta,
            (float*)b.data, b.rows));

        CUDAMatrix lz = *(m_vActivations.rbegin() + i + 1);
        CUDAMatrix w = *(m_vNabla_w.rbegin() + i);

        checkCuBlasErrors(cublasSgemm(m_hBlasHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            w.rows, w.cols, d.cols,
            &alpha,
            (const float*)d.data, d.rows,
            (const float*)lz.data, lz.rows,
            &beta,
            (float*)w.data, w.rows));
    }
}

void util::NeuralEx::Update()
{
    // 更新biases
    for (size_t i = 0; i < m_vBiases.size(); i++) {
        CUDAMatrix nb = m_vNabla_b[i];
        CUDAMatrix b = m_vBiases[i];

        void* update[] =
        {
            (void*)&nb.data, (void*)&nb.width,
            (void*)&b.data, (void*)&b.width,
            (void*)&m_fEta, (void*)&m_nBatch,
        };

        checkCudaErrors(cuLaunchKernel(m_fUpdate,
            b.height, 1, 1, b.width, 1, 1,
            0, nullptr, &update[0], 0));
    }

    // 更新weights
    for (size_t i = 0; i < m_vWeights.size(); i++) {
        CUDAMatrix nw = m_vNabla_w[i];
        CUDAMatrix w = m_vWeights[i];

        void* update[] =
        {
            (void*)&nw.data, (void*)&nw.width,
            (void*)&w.data, (void*)&w.width,
            (void*)&m_fEta, (void*)&m_nBatch,
        };

        checkCudaErrors(cuLaunchKernel(m_fUpdate,
            w.height, 1, 1, w.width, 1, 1,
            0, nullptr, &update[0], 0));
    }
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

    DestroyCUDAMatrix(m_nDeltaSum);
    DestroyCUDAMatrix(m_nSG);

    for (auto& item : m_vInputSum) {
        DestroyCUDAMatrix(item);
    }
    m_vInputSum.clear();

    DestroyCUDAMatrix(m_mTarget);

    if (m_hBlasHandle != nullptr) {
        checkCuBlasErrors(cublasDestroy(m_hBlasHandle));
        m_hBlasHandle = nullptr;
    }
}
