#include "neural.h"

using namespace Eigen;

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

    // ��������
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

    // �����������
    // ����batch�ɱ�,Ĭ���Ƚ���batch=1
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

    // �������ά��
    if (m_vActivations[0].rows() != in.rows()) {
        return false;
    }

    // ������ά��
    if (m_mTarget.rows() != target.rows()) {
        return false;
    }

    // ����ͬ
    if (m_nBatch == batch) {
        m_vActivations[0] = in;
        m_mTarget = target;
        return true;
    }

    // ����ͬ
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

    // ��ʧ����:loss = 1/2*(t - E)^2
    MatrixXf E = 1.0f / 2.0f * (target.array() - out.array()).pow(2);
    loss = E.sum() / out.cols();

    return true;
}

void util::Neural::SGD()
{
    // ǰ�򴫲�
    FeedForward();

    // �������
    BackProp();

    // batch���������²���
    Update();
}

void util::Neural::FeedForward()
{
    // ȫ��·�㼶
    size_t len = m_vActivations.size();

    // m_vActivations��һ��Ϊ�����
    // ÿ��ʹ��S�ͼ����
    // S�ͼ����:f(x) = 1/(1+e^(-x))
    // S�ͼ��������:f(x)' = f(x)*(1 - f(x))
    for (size_t i = 0; i < len - 1; i++) {
        m_vInputSum[i] = (m_vWeights[i] * m_vActivations[i]).colwise() + m_vBiases[i];
        m_vActivations[i + 1] = 1.0f / (1.0f + (-m_vInputSum[i]).array().exp());
    }
}

void util::Neural::BackProp()
{
    // ��Ŀ�����
    // ����:loss = 1/2*(t - E)^2
    // ��������:loss' = E - t
    const MatrixXf& z = m_vActivations.back();
    MatrixXf sg = z.array() * (1.0f - z.array());
    MatrixXf delta = (z - m_mTarget).array() * sg.array();
    m_vNabla_b.back() = delta.rowwise().sum();

    const MatrixXf& lz = *(m_vActivations.rbegin() + 1);
    m_vNabla_w.back() = delta * lz.transpose();

    // backprop���򴫲�
    size_t len = m_vActivations.size();
    for (size_t i = 1; i < len - 1; i++) {
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
    // ����biases
    for (size_t i = 0; i < m_vBiases.size(); i++) {
        m_vBiases[i] -= (m_fEta * m_vNabla_b[i] / m_nBatch);
    }

    // ����weights
    for (size_t i = 0; i < m_vWeights.size(); i++) {
        m_vWeights[i] -= (m_fEta * m_vNabla_w[i] / m_nBatch);
    }
}