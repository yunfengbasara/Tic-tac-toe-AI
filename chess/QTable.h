#pragma once
#include <array>
#include <map>
#include "TicRule.h"
#include "../Eigen/Core"

namespace chess
{
	class QTable
	{
	public:
		QTable();
		~QTable();

		// 生成Q表
		void Create();

		void Print();

	private:
		// Q表存储规则:
		// 代表该布局下,每个空位落子CROSS后的得分
		// 如果需要计算落子CIRCLE的得分,则需要反转棋子
		std::map<std::array<int, 9>, Eigen::Matrix3f> m_mStore;

		// 井字棋逻辑
		Tic m_nRule;
	};
}