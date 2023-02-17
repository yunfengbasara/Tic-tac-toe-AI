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

	public:
		// 批处理大小
		int m_nBatch = 0;

		// 学习速率
		float m_fEta = 0.3;

		// 每层节点个数
		std::vector<int> m_vNetParam;

		// 比较目标
		Eigen::MatrixXf	m_mTarget;

		// 和所有层对应
		// 每层激活输出(默认第一层输入)
		std::vector<Eigen::MatrixXf> m_vActivations;

		// 从隐藏层开始
		// 每层输入和
		std::vector<Eigen::MatrixXf> m_vInputSum;

		// 每层偏移
		std::vector<Eigen::VectorXf> m_vBiases;

		// 偏移临时记录
		std::vector<Eigen::VectorXf> m_vNabla_b;

		// 每层节点权重
		// 隐含层l为行,l-1为列,每行代表该节点所有输入
		// 这里有个性质:转置后,每行代表该节点所有输出
		std::vector<Eigen::MatrixXf> m_vWeights;

		// 权重临时记录
		std::vector<Eigen::MatrixXf> m_vNabla_w;
	};
}