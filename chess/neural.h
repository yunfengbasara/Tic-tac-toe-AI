#pragma once
#include <vector>
#include <string>
#include "../Eigen/Core"

namespace util
{
	class Neural 
	{
	public:
		Neural();
		~Neural();

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

	public:
		// �������С
		int m_nBatch = 0;

		// ѧϰ����
		float m_fEta = 0.3;

		// ÿ��ڵ����
		std::vector<int> m_vNetParam;

		// �Ƚ�Ŀ��
		Eigen::MatrixXf	m_mTarget;

		// �����в��Ӧ
		// ÿ�㼤�����(Ĭ�ϵ�һ������)
		std::vector<Eigen::MatrixXf> m_vActivations;

		// �����ز㿪ʼ
		// ÿ�������
		std::vector<Eigen::MatrixXf> m_vInputSum;

		// ÿ��ƫ��
		std::vector<Eigen::VectorXf> m_vBiases;

		// ƫ����ʱ��¼
		std::vector<Eigen::VectorXf> m_vNabla_b;

		// ÿ��ڵ�Ȩ��
		// ������lΪ��,l-1Ϊ��,ÿ�д���ýڵ���������
		// �����и�����:ת�ú�,ÿ�д���ýڵ��������
		std::vector<Eigen::MatrixXf> m_vWeights;

		// Ȩ����ʱ��¼
		std::vector<Eigen::MatrixXf> m_vNabla_w;
	};
}