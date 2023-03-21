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
		// Q表
		typedef std::map<std::array<int, 9>, Eigen::Matrix3f> QTABLE;

		// 用于记录上一轮的状态
		typedef struct {
			float reward;
			int idxpos;
			QTABLE::iterator pItem;
		}STATUS;

		// 生成Q表
		void Create();

		void Print();

	private:
		void UpdateQTable(STATUS& st, float maxvalue, bool print = false);

	private:

		// Q表存储规则:
		// 代表该布局下,每个空位落子CROSS后的得分
		// 如果需要计算落子CIRCLE的得分,则需要反转棋子
		QTABLE m_mStore;

		// 井字棋逻辑
		Tic m_nRule;

		// 默认CROSS首先落子
		Tic::RoleType m_nRole = Tic::RoleType::CROSS;
		
		// 上一轮CROSS的状态
		STATUS	m_nCrossInfo;

		// 上一轮CIRCLE的状态
		STATUS	m_nCircleInfo;
	};
}