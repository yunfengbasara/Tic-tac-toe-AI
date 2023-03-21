#pragma once
#include <vector>
#include <array>
#include "../Eigen/Core"

namespace chess
{
	class Tic
	{
	public:
		enum RoleType {
			CROSS = 1,	// x
			CIRCLE = 2,	// o
		};

		enum GameType {
			UNOVER = 0,
			CROSSW = 1,
			CIRCLEW = 2,
			DRAW	= 3,
		};

		Tic();
		~Tic();

		const Eigen::Matrix3i& Board();
		const Eigen::Matrix3i& RBoard();

		// 重置
		void Reset();

		// 随机下一个位置
		uint16_t RandomPos();

		// 按顺序下一个位置
		uint16_t NextPos();

		// 获取当前盘面最大值
		float GetMaxScore(const Eigen::Matrix3f& score,  
			int& row, int& col);

		// 创建当前局面初始化分数
		Eigen::Matrix3f CreateValue(float score);

		// 创建随机对局
		bool Create(const std::vector<int>& steps, 
			Eigen::Matrix3i& board,
			GameType& type, int& lp);

		// 在索引位置放下role棋子
		bool Turn(RoleType role, int idx);

		// 撤销该位置的棋子
		bool Revoke(int idx);

		// 检查本局是否结束
		GameType Check(int idx);

		// 反转棋子
		void Reverse();

	private:
		// 空位置数量
		uint16_t		m_nEmpCnt;

		// 棋盘索引 1:可下位置 0:不可下位置
		uint16_t		m_nIndex;

		// 棋盘数据 0:null 1:x 2:o
		Eigen::Matrix3i m_nBoard;

		// 反转棋盘,便于快速获取反转棋盘
		Eigen::Matrix3i	m_nRBoard;
	};
}