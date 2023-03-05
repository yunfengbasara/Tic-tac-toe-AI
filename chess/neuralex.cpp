#include "neuralex.h"
#include "util.h"
#include <Windows.h>
#include <chrono>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

using namespace Eigen;
using namespace util;
using namespace std;
using namespace std::chrono;


bool util::CUDAMatrix::Create(const HOSTMatrix& hostm)
{
    cols = hostm.cols();
    rows = hostm.rows();
    stride = cols * FZ;
    
    cudaError_t st = cudaMallocPitch(
        &data, &pitch, stride, rows);

    pitchcols = pitch / FZ;
    size = stride * rows;

    if (st != cudaSuccess) {
        return false;
    }

	return true;
}

bool util::CUDAMatrix::Destroy()
{
    cudaError_t st = cudaFree(data);
    if (st != cudaSuccess) {
        return false;
    }

    *this = CUDAMatrix();

	return true;
}

util::NeuralEx::NeuralEx()
{
    HOSTMatrix ma(3600, 777);
    ma.setRandom();

    CUDAMatrix cuma;
    cuma.Create(ma);

    HOSTMatrix mb(3600, 1);

    CUDAMatrix cumc;
    cumc.Create(mb);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMemcpy2DAsync(cuma.data, cuma.pitch,
        ma.data(), ma.cols() * FZ,
        ma.cols() * FZ, ma.rows(),
        cudaMemcpyHostToDevice, stream);

    //matrixSumCuda2 << <cuma.h, cuma.w, cuma.w * FZ, stream >> > (cuma, cumc);

    cudaError_t err = cudaStreamSynchronize(stream);

    VectorXf mc2(cumc.rows);
    cudaMemcpy2DAsync(mc2.data(), mc2.cols() * FZ,
        cumc.data, cumc.pitch,
        mc2.cols() * FZ, mc2.rows(),
        cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cuma.Destroy();
    cumc.Destroy();

    //bool ok = true;
    //float eps = 1.e-4;
    //const auto& c1 = mc1.array();
    //const auto& c2 = mc2.array();
    //for (int i = 0; i < mc1.size(); i++) {
    //    if (fabs(c1(i) - c2(i)) > eps) {
    //        cout << i << endl;
    //        cout << "-------" << endl;
    //        cout << c1(i) << endl;
    //        cout << "-------" << endl;
    //        cout << c2(i) << endl;
    //        ok = false;
    //        break;
    //    }
    //}

    //if (ok) {
    //    cout << "PASS";
    //}
    //else {
    //    cout << "WRONG!!!";
    //}
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
        item.Destroy();
    }
    m_vBiases.clear();

    for (auto& item : m_vNabla_b) {
        item.Destroy();
    }
    m_vNabla_b.clear();

    for (auto& item : m_vWeights) {
        item.Destroy();
    }
    m_vWeights.clear();

    for (auto& item : m_vNabla_w) {
        item.Destroy();
    }
    m_vNabla_w.clear();

    for (auto& item : m_vActivations) {
        item.Destroy();
    }
    m_vActivations.clear();

    for (auto& item : m_vInputSum) {
        item.Destroy();
    }
    m_vInputSum.clear();

    m_mTarget.Destroy();
}
