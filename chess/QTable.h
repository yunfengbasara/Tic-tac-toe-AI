#pragma once
#include <array>
#include <map>
#include "../Eigen/Core"

namespace chess
{
	class QTable
	{
	public:
		QTable();
		~QTable();

	private:
		// 计算本轮得分
		Eigen::Matrix3f Reward(const Eigen::Matrix3i& st);

	private:
		// Q表存储规则:
		// 按照落子CROSS来计算得分
		// 如果当前轮到CIRCLE，则需要反转棋子
		std::map<Eigen::Matrix3i, Eigen::Matrix3f> Store;
	};
}