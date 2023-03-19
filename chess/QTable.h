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

		// ����Q��
		void Create();

		void Print();

	private:
		// Q��洢����:
		// ����ò�����,ÿ����λ����CROSS��ĵ÷�
		// �����Ҫ��������CIRCLE�ĵ÷�,����Ҫ��ת����
		std::map<std::array<int, 9>, Eigen::Matrix3f> m_mStore;

		// �������߼�
		Tic m_nRule;
	};
}