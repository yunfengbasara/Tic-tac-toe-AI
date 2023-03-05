#pragma once
#include <vector>
#include <string>
#include "../Eigen/Core"

namespace util
{
	// �����������float����
	#define FZ	sizeof(float)

	// ����CUDA�еľ���������������,��˶��尴�����ȵ�CPU����
	typedef Eigen::Matrix<
		float, 
		Eigen::Dynamic, 
		Eigen::Dynamic, 
		Eigen::RowMajor> HOSTMatrix;

	// ����CUDA���õ��ľ���ṹ
	struct CUDAMatrix {
		size_t rows			=	0;		// ��������,height
		size_t cols			=	0;		// ��������,width
		size_t stride		=	0;		// һ���ֽڴ�С:cols * FZ
		size_t size			=	0;		// ����ȫ���ֽڴ�С:stride * rows
		size_t pitch		=	0;		// ����cuda������һ���ֽڴ�С
		size_t pitchcols	=	0;		// ����cuda������һ��Ԫ�ظ���:pitch / FZ
		float* data			=	nullptr;// ��������

		bool Create(const HOSTMatrix& hostm);
		bool Destroy();
	};

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