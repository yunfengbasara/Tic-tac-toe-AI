#include "QTable.h"
#include "TicRule.h"
#include <random>
#include <iostream>

using namespace Eigen;
using namespace chess;

QTable::QTable()
{
}

QTable::~QTable()
{
}

void QTable::Create()
{
	// ���Թ̶�
	auto seed = std::default_random_engine(std::random_device()());
	std::vector<int> steps = { 0,1,2,3,4,5,6,7,8 };
	std::shuffle(steps.begin(), steps.end(), seed);

	std::array<int, 9> bd;
	for (int i = 0; i < 5; i++) {
		m_nRule.Reset();
		float r = 0.0f;
		int idx = -1;
		Tic::GameType t = Tic::GameType::UNOVER;
		// ��һ��״̬��Q������
		std::map<std::array<int, 9>, Eigen::Matrix3f>::iterator lit;

		for (int st = 0; true; st++) {
			// ����ÿ�ν�����ɫ
			// �õ�ǰ״̬������һ�ֵ�Q��
			float maxvalue = 0;

			const Matrix3i& board = m_nRule.Board();
			memcpy(&bd[0], board.data(), 9 * sizeof(int));

			auto it = m_mStore.find(bd);
			if (it != m_mStore.end()) {
				maxvalue = it->second.maxCoeff();
			}

			// ����Q��
			if (idx != -1) {
				auto& v = lit->second(idx / 3, idx % 3);
				v = (1 - 0.27) * v + 0.27 * (r + 0.9 * maxvalue);

				Matrix3i a(lit->first.data());
				std::cout << a << std::endl;
				Matrix3f b(lit->second.data());
				std::cout << b << std::endl;
				std::cout << "-----" << std::endl;
			}

			// ��Ϸ����
			if (t != Tic::GameType::UNOVER) {
				break;
			}

			// ����ѡ�е�����λ��
			// ÿ�δ�Q����ѡ��һ����õ�λ��
			// ���Q��û�м�¼,�����ѡ��һ��λ��
			bool useRandom = (std::random_device()() % 10) < 0;

			if (it == m_mStore.end() || useRandom) {
				//idx = m_nRule.RandomPos();
				idx = steps[st];
				// ��ʼ��Ӧ������һ���Ƚ�С����ֵ���������Ϊ0����Զ���������
				// ���ߵ�λ������ΪС��ֵ��������λ����Ϊ0
				it = m_mStore.insert({ bd, m_nRule.CreateValue(0.1) }).first;
			}
			else {
				idx = steps[st];
				//idx = m_nRule.MaxScorePos(it->second);
			}

			// ���±���Q������
			lit = it;

			// �����λ�õ�reward
			// ÿ�ζ���CROSS�ķ����ߵ�
			// ������CIRCLEW�����
			m_nRule.Turn(Tic::RoleType::CROSS, idx);
			t = m_nRule.Check(idx);

			if (t == Tic::GameType::UNOVER) {
				r = 0.0f;
			}
			else if (t == Tic::GameType::DRAW) {
				r = 0.1f;
			}
			else if (t == Tic::GameType::CROSSW) {
				r = 1.0f;
			}

			// ��������,ʹ��ÿ�����Ӷ���CROSS
			m_nRule.Reverse();
		}

	}
}

void QTable::Print()
{
	//for (auto [k, v] : m_mStore) {
	//	Matrix3i board(k.data());
	//	std::cout << board << std::endl;
	//	Matrix3f value(v.data());
	//	std::cout << value << std::endl;
	//	std::cout << "-----" << std::endl;
	//}

	//Matrix3i board;
	//board <<
	//	0, 0, 0,
	//	0, 0, 0,
	//	0, 0, 0;

	//std::array<int, 9> bd;
	//memcpy(&bd[0], board.data(), 9 * sizeof(int));
	//auto it = m_mStore.find(bd);
	//if (it != m_mStore.end()) {
	//	Matrix3f value(it->second.data());
	//	std::cout << value << std::endl;
	//}
}
