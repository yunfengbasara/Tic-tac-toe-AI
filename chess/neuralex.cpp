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

    // ��������
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

    // �����������
    // ����batch�ɱ�,Ĭ���Ƚ���batch=1
    m_nBatch = 1;

    for (size_t i = 0; i < p.size(); i++) {
        //CUDAMatrix cm = CreateCUDAMatrix(p[i], m_nBatch);
        HOSTMatrix hm = HOSTMatrix::Random(p[i], m_nBatch);
        CUDAMatrix cm = CreateCUDAMatrix(hm);
        m_vActivations.push_back(cm);
    }

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

    // �������ά��
    if (m_vActivations[0].height != in.rows()) {
        return false;
    }

    // ������ά��
    if (m_mTarget.height != target.rows()) {
        return false;
    }

    // ����ͬ
    if (m_nBatch == batch) {
        CopyHostToCUDA(in, m_vActivations[0]);
        CopyHostToCUDA(target, m_mTarget);
        return true;
    }

    // ����ͬ ���¹����������ܺͲ�
    m_nBatch = batch;

    // ���������
    for (auto& item : m_vActivations) {
        DestroyCUDAMatrix(item);
    }
    m_vActivations.clear();

    for (size_t i = 0; i < m_vNetParam.size(); i++) {
        CUDAMatrix cm = CreateCUDAMatrix(m_vNetParam[i], m_nBatch);
        m_vActivations.push_back(cm);
    }

    // ��������Ͳ�
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
    // ǰ�򴫲�
    FeedForward();

    // �������
    BackProp();

    // batch���������²���
    Update();
}

void util::NeuralEx::FeedForward()
{
    // m_vActivations��һ��Ϊ�����
    // ÿ��ʹ��S�ͼ����
    // S�ͼ����:f(x) = 1/(1+e^(-x))
    // S�ͼ��������:f(x)' = f(x)*(1 - f(x))
    for (size_t i = 0; i < m_vNetParam.size() - 1; i++) {
        int width = m_vInputSum[i].width;
        int height = m_vInputSum[i].height;

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        int gx = (width + block.x - 1) / block.x;
        int gy = (height + block.y - 1) / block.y;

        // �����Ȩ�س˻�
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

        // ����ƫ��
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

        // ����
        void* actparams[] =
        {
            (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].pitchwidth,
            (void*)&m_vActivations[i + 1].data, (void*)&m_vActivations[i + 1].pitchwidth
        };

        checkCudaErrors(cuLaunchKernel(m_fActivation,
            height, 1, 1, width, 1, 1,
            0, m_nStream, &actparams[0], 0));
        checkCudaErrors(cuStreamSynchronize(m_nStream));

        // ���㰴�����
        //void* params1[] = 
        //{
        //    (void*)&m_vInputMulTmp[i].data, (void*)&m_vInputMulTmp[i].pitchwidth,
        //    (void*)&m_vInputSum[i].data, (void*)&m_vInputSum[i].pitchwidth
        //};

        //checkCudaErrors(cuLaunchKernel(m_fReduction,
        //    height, 1, 1,
        //    width, 1, 1,
        //    m_vInputSum[i].stride, m_nStream,
        //    &params1[0], 0));
        //checkCudaErrors(cuStreamSynchronize(m_nStream));

        //HOSTMatrix bm(m_vInputMulTmp[i].height, m_vInputMulTmp[i].width);
        //CopyCUDAToHost(m_vInputMulTmp[i], bm, m_nStream);

        //HOSTMatrix am(m_vBiases[i].height, m_vBiases[i].width);
        //CopyCUDAToHost(m_vBiases[i], am, m_nStream);

        //HOSTMatrix mc1 = bm.colwise() + am.col(0);
        //HOSTMatrix mc2(m_vInputSum[i].height, m_vInputSum[i].width);
        //CopyCUDAToHost(m_vInputSum[i], mc2, m_nStream);

        //float eps = 1.e-4;
        //const auto& c1 = mc1.array();
        //const auto& c2 = mc2.array();
        //for (int i = 0; i < mc1.size(); i++) {
        //    if (fabs(c1(i) - c2(i)) > eps) {
        //        cout << i << endl;
        //        cout << c1(i) << endl;
        //        cout << c2(i) << endl;
        //        break;
        //    }
        //}
    }
}

void util::NeuralEx::BackProp()
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
