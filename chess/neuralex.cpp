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

    CUmodule module;
    if (!LoadCUBIN(cubin, module)) {
        return;
    }

    HOSTMatrix ma(3667, 366);
    ma.setRandom();

    CUDAMatrix cuma;
    CreateCUDAMatrix(ma, cuma);

    HOSTMatrix mb(3667, 1);

    CUDAMatrix cumc;
    CreateCUDAMatrix(mb, cumc);

    CUresult ret;
    const char* errorStr = NULL;

    CUstream stream;
    ret = cuStreamCreate(&stream, cudaStreamNonBlocking);
    cuGetErrorString(ret, &errorStr);

    CopyHostToCUDA(ma, cuma, stream);

    CUfunction kernel_addr;
    if (cuModuleGetFunction(&kernel_addr, module, 
            "reduction") != CUDA_SUCCESS) {
        return;
    }

    void* arr[] = { 
        (void*)&cuma.data, (void*)&cuma.pitchcols, 
        (void*)&cumc.data, (void*)&cumc.pitchcols
    };
    ret = cuLaunchKernel(kernel_addr,
        cuma.height, 1, 1,
        cuma.width, 1, 1,
        cuma.stride, stream,
        &arr[0], 0);
    cuGetErrorString(ret, &errorStr);
    
    ret = cuStreamSynchronize(stream);
    cuGetErrorString(ret, &errorStr);

    HOSTMatrix mc2(cumc.height, 1);

    CopyCUDAToHost(cumc, mc2, stream);

    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    DestroyCUDAMatrix(cuma);
    DestroyCUDAMatrix(cumc);

    VectorXf mc1 = ma.rowwise().sum();

    bool ok = true;
    float eps = 1.e-4;
    const auto& c1 = mc1.array();
    const auto& c2 = mc2.array();
    for (int i = 0; i < mc1.size(); i++) {
        if (fabs(c1(i) - c2(i)) > eps) {
            cout << i << endl;
            cout << "-------" << endl;
            cout << c1(i) << endl;
            cout << "-------" << endl;
            cout << c2(i) << endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        cout << "PASS";
    }
    else {
        cout << "WRONG!!!";
    }
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

    m_vNetParam = p;

    //// 构建网络
    //for (size_t i = 1; i < p.size(); i++) {
    //    VectorXf x = VectorXf::Random(p[i]);
    //    m_vBiases.push_back(x);
    //}

    //for (size_t i = 1; i < p.size(); i++) {
    //    VectorXf x(p[i]);
    //    x.setZero();
    //    m_vNabla_b.push_back(x);
    //}

    //for (size_t i = 1; i < p.size(); i++) {
    //    MatrixXf x = MatrixXf::Random(p[i], p[i - 1]);
    //    m_vWeights.push_back(x);
    //}

    //for (size_t i = 1; i < p.size(); i++) {
    //    MatrixXf x(p[i], p[i - 1]);
    //    x.setZero();
    //    m_vNabla_w.push_back(x);
    //}

    //// 构建输入输出
    //// 由于batch可变,默认先建立batch=1
    //m_nBatch = 1;

    //for (size_t i = 0; i < p.size(); i++) {
    //    MatrixXf x = MatrixXf(p[i], m_nBatch);
    //    m_vActivations.push_back(x);
    //}

    //for (size_t i = 1; i < p.size(); i++) {
    //    MatrixXf x = MatrixXf(p[i], m_nBatch);
    //    m_vInputSum.push_back(x);
    //}

    //m_mTarget = MatrixXf(p.back(), m_nBatch);
    return true;
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
}
