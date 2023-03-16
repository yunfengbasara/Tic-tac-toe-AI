#pragma once
#include <vector>
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

		void Reset();
		int RandomPos();
		bool Create(const std::vector<int>& steps, 
			Eigen::Matrix3i& board,
			GameType& type, int& lp);

	private:
		bool Turn(RoleType role, int idx);
		GameType Check(int idx);

	private:
		// 当前轮到的棋子
		RoleType		m_nRole;

		// 空位置数量
		uint16_t		m_nEmpCnt;

		// 棋盘索引 1:可下位置 0:不可下位置
		uint16_t		m_nIndex;

		// 棋盘数据 0:null 1:x 2:o
		Eigen::Matrix3i m_nBoard;
	};
}