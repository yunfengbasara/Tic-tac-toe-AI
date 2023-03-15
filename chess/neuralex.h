#pragma once
#include <vector>
#include <string>
#include "type.h"
#include "cuda.h"
#include "cublas_v2.h"

namespace util
{
	class NeuralEx
	{
	public:
		NeuralEx();
		~NeuralEx();

		// ��ʧ����
		enum CostType {
			Quadratic = 0,		//	1 / 2 * (t - E) ^ 2
			CrossEntropy = 1,	//	-ylna - (1-y)ln(1 - a)
		};

		// ��ʼ��
		bool InitBuild(std::vector<int> p);

		// ��������
		bool SetSample(
			HOSTMatrix& in,
			HOSTMatrix& target
		);

		// �����ȶ�
		bool CompareSample(
			HOSTMatrix& in,
			HOSTMatrix& target,
			HOSTMatrix& out,
			float& loss
		);

		void SetLearnRate(float eta);
		void SetRegularization(float lambda);
		void SetCostType(CostType type);
		void SetTotalItem(int count);

		// ����ݶ��½�
		void SGD();

		// ����
		bool Save(const std::wstring& path);

		// ��ȡ
		bool Load(const std::wstring& path);

	private:
		// ǰ�����
		void FeedForward();

		// ���򴫲�(Ŀ��)
		void BackPropLast();

		// ���򴫲���
		void BackPropLine();

		// ����Ȩֵ
		void Update();

		// �ͷ��ڴ�
		void Release();
	public:
		// �������С
		int m_nBatch = 0;

		// ѧϰ����
		float m_fEta = 0.3;

		// regularization����
		float m_fLambda = 5.0;

		// ѵ�����ݴ�С
		int m_nTotalItem = 0;

		// ��ʧ��������
		CostType m_nCost = Quadratic;

		// ÿ��ڵ����
		std::vector<int> m_vNetParam;

		// �Ƚ�Ŀ��
		CUDAMatrix	m_mTarget;

		// �����в��Ӧ
		// ÿ�㼤�����(Ĭ�ϵ�һ������)
		std::vector<CUDAMatrix> m_vActivations;

		// �����ز㿪ʼ
		// ÿ�������
		std::vector<CUDAMatrix> m_vInputSum;

		// ÿ��ƫ�� w = 1
		std::vector<CUDAMatrix> m_vBiases;

		// ƫ����ʱ��¼ w = 1
		std::vector<CUDAMatrix> m_vNabla_b;

		// ÿ��ڵ�Ȩ��
		// ������lΪ��,l-1Ϊ��,ÿ�д���ýڵ���������
		// �����и�����:ת�ú�,ÿ�д���ýڵ��������
		std::vector<CUDAMatrix> m_vWeights;

		// Ȩ����ʱ��¼
		std::vector<CUDAMatrix> m_vNabla_w;

		// cubinģ��
		CUmodule m_nModule = nullptr;

		// cublas
		cublasHandle_t m_hBlasHandle = nullptr;

		// cuda����
		CUfunction m_fColwiseAdd = nullptr;
		CUfunction m_fActivation = nullptr;
		CUfunction m_fActivatePrime = nullptr;
		CUfunction m_fDeltaQuadratic = nullptr;
		CUfunction m_fDeltaCrossEntropy = nullptr;
		CUfunction m_fArrayMul = nullptr;
		CUfunction m_fUpdateBias = nullptr;
		CUfunction m_fUpdateWeight = nullptr;

		// Ŀ��ƫ����ʱ������Quadratic������
		CUDAMatrix m_nSG;

		// ƫ����ʱ�������
		CUDAMatrix m_nDeltaSum;

		// ƫ����ʱ����
		std::vector<CUDAMatrix> m_vDelta;
	};
}