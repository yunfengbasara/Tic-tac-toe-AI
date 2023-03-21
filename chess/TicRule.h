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

		// ����
		void Reset();

		// �����һ��λ��
		uint16_t RandomPos();

		// ��˳����һ��λ��
		uint16_t NextPos();

		// ��ȡ��ǰ�������ֵ
		float GetMaxScore(const Eigen::Matrix3f& score,  
			int& row, int& col);

		// ������ǰ�����ʼ������
		Eigen::Matrix3f CreateValue(float score);

		// ��������Ծ�
		bool Create(const std::vector<int>& steps, 
			Eigen::Matrix3i& board,
			GameType& type, int& lp);

		// ������λ�÷���role����
		bool Turn(RoleType role, int idx);

		// ������λ�õ�����
		bool Revoke(int idx);

		// ��鱾���Ƿ����
		GameType Check(int idx);

		// ��ת����
		void Reverse();

	private:
		// ��λ������
		uint16_t		m_nEmpCnt;

		// �������� 1:����λ�� 0:������λ��
		uint16_t		m_nIndex;

		// �������� 0:null 1:x 2:o
		Eigen::Matrix3i m_nBoard;

		// ��ת����,���ڿ��ٻ�ȡ��ת����
		Eigen::Matrix3i	m_nRBoard;
	};
}