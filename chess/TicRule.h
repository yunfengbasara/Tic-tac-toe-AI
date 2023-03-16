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
		// ��ǰ�ֵ�������
		RoleType		m_nRole;

		// ��λ������
		uint16_t		m_nEmpCnt;

		// �������� 1:����λ�� 0:������λ��
		uint16_t		m_nIndex;

		// �������� 0:null 1:x 2:o
		Eigen::Matrix3i m_nBoard;
	};
}