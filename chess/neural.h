#pragma once
#include <vector>
#include "../Eigen/Dense"

namespace util
{
	class Neural 
	{
	public:
		Neural();
		~Neural();

		// ��ʼ��(����һ��)
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

		// ����ݶ��½�
		void SGD();

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
		float m_fEta = 0.03;

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