#pragma once
#include <vector>
#include <string>
#include "../Eigen/Core"

namespace util
{
	// 整个网络采用float类型
	#define FZ	sizeof(float)

	// 由于CUDA中的矩阵按照行优先排列,因此定义按行优先的CPU矩阵
	typedef Eigen::Matrix<
		float, 
		Eigen::Dynamic, 
		Eigen::Dynamic, 
		Eigen::RowMajor> HOSTMatrix;

	// 定义CUDA中用到的矩阵结构
	struct CUDAMatrix {
		size_t rows			=	0;		// 矩阵行数,height
		size_t cols			=	0;		// 矩阵列数,width
		size_t stride		=	0;		// 一行字节大小:cols * FZ
		size_t size			=	0;		// 矩阵全部字节大小:stride * rows
		size_t pitch		=	0;		// 经过cuda对齐后的一行字节大小
		size_t pitchcols	=	0;		// 经过cuda对齐后的一行元素个数:pitch / FZ
		float* data			=	nullptr;// 矩阵数据

		bool Create(const HOSTMatrix& hostm);
		bool Destroy();
	};

	class NeuralEx
	{
	public:
		NeuralEx();
		~NeuralEx();

		// 初始化
		bool InitBuild(std::vector<int> p);

		// 样本输入
		bool SetSample(
			Eigen::MatrixXf& in,
			Eigen::MatrixXf& target
		);

		// 样本比对
		bool CompareSample(
			Eigen::MatrixXf& in,
			Eigen::MatrixXf& target,
			Eigen::MatrixXf& out,
			float& loss
		);

		void SetLearnRate(float eta);

		// 随机梯度下降
		void SGD();

		// 保存
		bool Save(const std::wstring& path);

		// 读取
		bool Load(const std::wstring& path);

	private:
		// 前向计算
		void FeedForward();

		// 反向传播
		void BackProp();

		// 更新权值
		void Update();

		// 释放内存
		void Release();
	public:
		// 批处理大小
		int m_nBatch = 0;

		// 学习速率
		float m_fEta = 0.3;

		// 每层节点个数
		std::vector<int> m_vNetParam;

		// 比较目标
		CUDAMatrix	m_mTarget;

		// 和所有层对应
		// 每层激活输出(默认第一层输入)
		std::vector<CUDAMatrix> m_vActivations;

		// 从隐藏层开始
		// 每层输入和
		std::vector<CUDAMatrix> m_vInputSum;

		// 每层偏移 cols = 1
		std::vector<CUDAMatrix> m_vBiases;

		// 偏移临时记录 cols = 1
		std::vector<CUDAMatrix> m_vNabla_b;

		// 每层节点权重
		// 隐含层l为行,l-1为列,每行代表该节点所有输入
		// 这里有个性质:转置后,每行代表该节点所有输出
		std::vector<CUDAMatrix> m_vWeights;

		// 权重临时记录
		std::vector<CUDAMatrix> m_vNabla_w;
	};
}