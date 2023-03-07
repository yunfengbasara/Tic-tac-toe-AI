#pragma once
#include <vector>
#include <string>
#include "type.h"

namespace util
{
	class NeuralEx
	{
	public:
		NeuralEx();
		~NeuralEx();

		// ��ʼ��
		bool InitBuild(std::vector<int> p);

		// ��������
		bool SetSample(
			Eigen::MatrixXf& in,
			Eigen::MatrixXf& target
		);

		// �����ȶ�
		bool CompareSample(
			Eigen::MatrixXf& in,
			Eigen::MatrixXf& target,
			Eigen::MatrixXf& out,
			float& loss
		);

		void SetLearnRate(float eta);

		// ����ݶ��½�
		void SGD();

		// ����
		bool Save(const std::wstring& path);

		// ��ȡ
		bool Load(const std::wstring& path);

	private:
		// ǰ�����
		void FeedForward();

		// ���򴫲�
		void BackProp();

		// ����Ȩֵ
		void Update();

		// �ͷ��ڴ�
		void Release();
	public:
		// �������С
		int m_nBatch = 0;

		// ѧϰ����
		float m_fEta = 0.3;

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

		// ÿ��ƫ�� cols = 1
		std::vector<CUDAMatrix> m_vBiases;

		// ƫ����ʱ��¼ cols = 1
		std::vector<CUDAMatrix> m_vNabla_b;

		// ÿ��ڵ�Ȩ��
		// ������lΪ��,l-1Ϊ��,ÿ�д���ýڵ���������
		// �����и�����:ת�ú�,ÿ�д���ýڵ��������
		std::vector<CUDAMatrix> m_vWeights;

		// Ȩ����ʱ��¼
		std::vector<CUDAMatrix> m_vNabla_w;
	};
}