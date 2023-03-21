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
		// Q��
		typedef std::map<std::array<int, 9>, Eigen::Matrix3f> QTABLE;

		// ���ڼ�¼��һ�ֵ�״̬
		typedef struct {
			float reward;
			int idxpos;
			QTABLE::iterator pItem;
		}STATUS;

		// ����Q��
		void Create();

		void Print();

	private:
		void UpdateQTable(STATUS& st, float maxvalue, bool print = false);

	private:

		// Q��洢����:
		// ����ò�����,ÿ����λ����CROSS��ĵ÷�
		// �����Ҫ��������CIRCLE�ĵ÷�,����Ҫ��ת����
		QTABLE m_mStore;

		// �������߼�
		Tic m_nRule;

		// Ĭ��CROSS��������
		Tic::RoleType m_nRole = Tic::RoleType::CROSS;
		
		// ��һ��CROSS��״̬
		STATUS	m_nCrossInfo;

		// ��һ��CIRCLE��״̬
		STATUS	m_nCircleInfo;
	};
}