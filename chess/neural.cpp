#include "neural.h"
#include "util.h"
#include <Windows.h>

using namespace Eigen;
using namespace util;

util::Neural::Neural()
{
}

util::Neural::~Neural()
{
}

bool util::Neural::InitBuild(std::vector<int> p)
{
    if (p.size() < 3) {
        return false;
    }

    m_vBiases.clear();
    m_vNabla_b.clear();
    m_vWeights.clear();
    m_vNabla_w.clear();
    m_vActivations.clear();
    m_vInputSum.clear();

    m_vNetParam = p;

    // 构建网络
    for (size_t i = 1; i < p.size(); i++) {
        VectorXf x = VectorXf::Random(p[i]);
        m_vBiases.push_back(x);
    }

    for (size_t i = 1; i < p.size(); i++) {
        VectorXf x(p[i]);
        x.setZero();
        m_vNabla_b.push_back(x);
    }

    for (size_t i = 1; i < p.size(); i++) {
        MatrixXf x = MatrixXf::Random(p[i], p[i - 1]);
        m_vWeights.push_back(x);
    }

    for (size_t i = 1; i < p.size(); i++) {
        MatrixXf x(p[i], p[i - 1]);
        x.setZero();
        m_vNabla_w.push_back(x);
    }

    // 构建输入输出
    // 由于batch可变,默认先建立batch=1
    m_nBatch = 1;

    for (size_t i = 0; i < p.size(); i++) {
        MatrixXf x = MatrixXf(p[i], m_nBatch);
        m_vActivations.push_back(x);
    }

    for (size_t i = 1; i < p.size(); i++) {
        MatrixXf x = MatrixXf(p[i], m_nBatch);
        m_vInputSum.push_back(x);
    }

    m_mTarget = MatrixXf(p.back(), m_nBatch);
	return true;
}

bool util::Neural::SetSample(
    Eigen::MatrixXf& in,
    Eigen::MatrixXf& target)
{
    if (in.cols() != target.cols()) {
        return false;
    }

    int batch = in.cols();

    // 检查输入维度
    if (m_vActivations[0].rows() != in.rows()) {
        return false;
    }

    // 检查输出维度
    if (m_mTarget.rows() != target.rows()) {
        return false;
    }

    // 批相同
    if (m_nBatch == batch) {
        m_vActivations[0] = in;
        m_mTarget = target;
        return true;
    }

    // 批不同
    m_nBatch = batch;
    for (size_t i = 1; i < m_vActivations.size(); i++) {
        m_vActivations[i].resize(NoChange, m_nBatch);
    }

    for (auto& mat : m_vInputSum) {
        mat.resize(NoChange, m_nBatch);
    }

    m_vActivations[0] = in;
    m_mTarget = target;
    return true;
}

bool util::Neural::CompareSample(
    Eigen::MatrixXf& in, 
    Eigen::MatrixXf& target, 
    Eigen::MatrixXf& out, 
    float& loss)
{
    if (!SetSample(in, target)) {
        return false;
    }

    FeedForward();

    out = m_vActivations.back();

    // 损失函数:loss = 1/2*(t - E)^2
    MatrixXf E = 1.0f / 2.0f * (target.array() - out.array()).pow(2);
    loss = E.sum() / out.cols();

    return true;
}

void util::Neural::SGD()
{
    // 前向传播
    FeedForward();

    // 反向矫正
    BackProp();

    // batch处理完后更新参数
    Update();
}

void util::Neural::FeedForward()
{
    // m_vActivations第一层为输入层
    // 每层使用S型激活函数
    // S型激活函数:f(x) = 1/(1+e^(-x))
    // S型激活函数导数:f(x)' = f(x)*(1 - f(x))
    for (size_t i = 0; i < m_vNetParam.size() - 1; i++) {
        m_vInputSum[i] = (m_vWeights[i] * m_vActivations[i]).colwise() + m_vBiases[i];
        m_vActivations[i + 1] = 1.0f / (1.0f + (-m_vInputSum[i]).array().exp());
    }
}

void util::Neural::BackProp()
{
    // 求目标误差
    // 误差函数:loss = 1/2*(t - E)^2
    // 误差函数导数:loss' = E - t
    const MatrixXf& z = m_vActivations.back();
    MatrixXf sg = z.array() * (1.0f - z.array());
    MatrixXf delta = (z - m_mTarget).array() * sg.array();
    m_vNabla_b.back() = delta.rowwise().sum();

    const MatrixXf& lz = *(m_vActivations.rbegin() + 1);
    m_vNabla_w.back() = delta * lz.transpose();

    // backprop反向传播
    for (size_t i = 1; i < m_vNetParam.size() - 1; i++) {
        const MatrixXf& z = *(m_vActivations.rbegin() + i);
        MatrixXf sg = z.array() * (1.0f - z.array());
        const MatrixXf& w = *(m_vWeights.rbegin() + i - 1);
        delta = (w.transpose() * delta).array() * sg.array();
        *(m_vNabla_b.rbegin() + i) = delta.rowwise().sum();

        const MatrixXf& lz = *(m_vActivations.rbegin() + i + 1);
        *(m_vNabla_w.rbegin() + i) = delta * lz.transpose();
    }
}

void util::Neural::Update()
{
    // 更新biases
    for (size_t i = 0; i < m_vBiases.size(); i++) {
        m_vBiases[i] -= (m_fEta * m_vNabla_b[i] / m_nBatch);
    }

    // 更新weights
    for (size_t i = 0; i < m_vWeights.size(); i++) {
        m_vWeights[i] -= (m_fEta * m_vNabla_w[i] / m_nBatch);
    }
}

bool util::Neural::Save(const std::wstring& path)
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
        filesize += b.size() * sizeof(float);
    }
    // Weights大小
    for (auto& w : m_vWeights) {
        filesize += w.size() * sizeof(float);
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
        DWORD len = b.size() * sizeof(float);
        memcpy(lpMem + startfile, b.data(), len);
        startfile += len;
    }

    for (auto& w : m_vWeights) {
        DWORD len = w.size() * sizeof(float);
        memcpy(lpMem + startfile, w.data(), len);
        startfile += len;
    }

    return true;
}

bool util::Neural::Load(const std::wstring& path)
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
        b = Map<VectorXf>(pf, b.size());
        startfile += b.size() * sizeof(float);
    }

    for (auto& w : m_vWeights) {
        float* pf = (float*)(lpMem + startfile);
        w = Map<MatrixXf>(pf, w.rows(), w.cols());
        startfile += w.size() * sizeof(float);
    }

    return true;
}