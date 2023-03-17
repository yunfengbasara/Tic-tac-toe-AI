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
		// ���㱾�ֵ÷�
		Eigen::Matrix3f Reward(const Eigen::Matrix3i& st);

	private:
		// Q��洢����:
		// ��������CROSS������÷�
		// �����ǰ�ֵ�CIRCLE������Ҫ��ת����
		std::map<Eigen::Matrix3i, Eigen::Matrix3f> Store;
	};
}