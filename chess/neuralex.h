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

		// 损失函数
		enum CostType {
			Quadratic = 0,		//	1 / 2 * (t - E) ^ 2
			CrossEntropy = 1,	//	-ylna - (1-y)ln(1 - a)
		};

		// 初始化
		bool InitBuild(std::vector<int> p);

		// 样本输入
		bool SetSample(
			HOSTMatrix& in,
			HOSTMatrix& target
		);

		// 样本比对
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

		// 随机梯度下降
		void SGD();

		// 保存
		bool Save(const std::wstring& path);

		// 读取
		bool Load(const std::wstring& path);

	private:
		// 前向计算
		void FeedForward();

		// 反向传播(目标)
		void BackPropLast();

		// 反向传播链
		void BackPropLine();

		// 更新权值
		void Update();

		// 释放内存
		void Release();
	public:
		// 批处理大小
		int m_nBatch = 0;

		// 学习速率
		float m_fEta = 0.3;

		// regularization参数
		float m_fLambda = 5.0;

		// 训练数据大小
		int m_nTotalItem = 0;

		// 损失函数类型
		CostType m_nCost = Quadratic;

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

		// 每层偏移 w = 1
		std::vector<CUDAMatrix> m_vBiases;

		// 偏移临时记录 w = 1
		std::vector<CUDAMatrix> m_vNabla_b;

		// 每层节点权重
		// 隐含层l为行,l-1为列,每行代表该节点所有输入
		// 这里有个性质:转置后,每行代表该节点所有输出
		std::vector<CUDAMatrix> m_vWeights;

		// 权重临时记录
		std::vector<CUDAMatrix> m_vNabla_w;

		// cubin模块
		CUmodule m_nModule = nullptr;

		// cublas
		cublasHandle_t m_hBlasHandle = nullptr;

		// cuda函数
		CUfunction m_fColwiseAdd = nullptr;
		CUfunction m_fActivation = nullptr;
		CUfunction m_fActivatePrime = nullptr;
		CUfunction m_fDeltaQuadratic = nullptr;
		CUfunction m_fDeltaCrossEntropy = nullptr;
		CUfunction m_fArrayMul = nullptr;
		CUfunction m_fUpdateBias = nullptr;
		CUfunction m_fUpdateWeight = nullptr;

		// 目标偏差临时变量（Quadratic方法）
		CUDAMatrix m_nSG;

		// 偏导临时变量求和
		CUDAMatrix m_nDeltaSum;

		// 偏导临时变量
		std::vector<CUDAMatrix> m_vDelta;
	};
}