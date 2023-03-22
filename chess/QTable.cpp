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
	int times = 70000;

	std::array<int, 9> bd;
	for (int i = 0; i <= times; i++) {
		Tic::GameType gameresult;
		gameresult = Tic::GameType::UNOVER;

		m_nRole = Tic::RoleType::CROSS;

		m_nCrossInfo.idxpos = -1;
		m_nCircleInfo.idxpos = -1;

		m_nRule.Reset();

		while (true) {
			// 由于每次交换角色
			// 用当前状态更新上一轮的Q表
			STATUS& lst = m_nRole == Tic::RoleType::CROSS ?
				m_nCrossInfo : m_nCircleInfo;

			const Matrix3i& board = m_nRule.Board();
			memcpy(&bd[0], board.data(), 9 * sizeof(int));

			// 本轮Q表最大值
			float maxvalue = 0;

			// 本轮Q表最大值的索引
			int mrow, mcol;

			// 本轮索引（可能选中了不可走的位置）
			auto pItem = m_mStore.find(bd);
			if (pItem != m_mStore.end()) {
				maxvalue = m_nRule.GetMaxScore(pItem->second, mrow, mcol);
			}

			// 选择本轮最佳位置
			int idxpos;

			// 如果是新局面,则插入新状态
			if (pItem == m_mStore.end()) {
				pItem = m_mStore.insert({ bd, Matrix3f::Zero() }).first;
				idxpos = m_nRule.RandomPos();
			}
			// 使用本轮Q值最大的位置
			else {
				idxpos = mrow * 3 + mcol;
			}

			// 本轮是否随机一个位置
			bool useRandom = (std::random_device()() % 100) < 35;
			if (useRandom) {
				idxpos = m_nRule.RandomPos();
			}

			// 计算该位置的reward
			// 每次都是CROSS的分数高低
			// 不存在CIRCLEW的情况
			m_nRule.Turn(Tic::RoleType::CROSS, idxpos);
			gameresult = m_nRule.Check(idxpos);

			// 本轮reward
			float reward = 0.0f;
			//if (gameresult == Tic::GameType::CROSSW) {
			//	reward = 1.0f;
			//}

			// 更新上一轮Q表
			UpdateQTable(lst, maxvalue);

			// 记录本次的结果，用于下次更新
			lst.reward = reward;
			lst.pItem = pItem;
			lst.idxpos = idxpos;

			// 游戏结束
			if (gameresult != Tic::GameType::UNOVER) {
				break;
			}

			// 交换棋子,使得每次落子都是CROSS
			m_nRule.Reverse();

			if (m_nRole == Tic::RoleType::CROSS) {
				m_nRole = Tic::RoleType::CIRCLE;
			}
			else {
				m_nRole = Tic::RoleType::CROSS;
			}
		}

		float foScore = 0.0f;
		float fxScore = 0.0f;

		// 平局对双方结果相同
		if (gameresult == Tic::GameType::DRAW) {
			foScore = -0.01f;
			fxScore = -0.01f;
		}
		// X胜
		else if (m_nRole == Tic::RoleType::CROSS) {
			foScore = -1.0f;
			fxScore = 1.0f;
		}
		// O胜
		else {
			foScore = 1.0f;
			fxScore = -1.0f;
		}

		UpdateQTable(m_nCircleInfo, foScore);
		UpdateQTable(m_nCrossInfo, fxScore);
	}
}

void QTable::UpdateQTable(STATUS& st, float maxvalue, bool print)
{
	if (st.idxpos == -1) {
		return;
	}

	float eta = 0.17;
	auto& v = st.pItem->second(st.idxpos / 3, st.idxpos % 3);
	v = (1 - eta) * v + eta * (st.reward + 0.8 * maxvalue);

	if (print) {
		Matrix3i board(st.pItem->first.data());
		std::cout << board << std::endl;
		Matrix3f value(st.pItem->second.data());
		std::cout << value << std::endl;
		std::cout << "-----" << std::endl;
	}
}

void QTable::Print()
{
	std::cout << "Q Table size " << m_mStore.size() << std::endl;

	// 测试的开始几步
	std::vector<int> steps = {4,5};

	int times = 1;
	for (int i = 0; i < times; i++) {
		std::array<int, 9> bd;

		Tic::GameType gameresult;
		gameresult = Tic::GameType::UNOVER;

		m_nRule.Reset();

		m_nRole = Tic::RoleType::CROSS;

		for (int k = 0; true; k++) {
			if (m_nRole == Tic::RoleType::CROSS) {
				const Matrix3i& board = m_nRule.Board();
				memcpy(&bd[0], board.data(), 9 * sizeof(int));
			}
			else {
				const Matrix3i& board = m_nRule.RBoard();
				memcpy(&bd[0], board.data(), 9 * sizeof(int));
			}

			Matrix3f value = Matrix3f::Zero();

			int idxpos;
			int mrow, mcol;
			auto pItem = m_mStore.find(bd);
			if (pItem != m_mStore.end()) {
				m_nRule.GetMaxScore(pItem->second, mrow, mcol);
				idxpos = mrow * 3 + mcol;
				value = Matrix3f(pItem->second.data());
			}
			else {
				idxpos = m_nRule.RandomPos();
			}

			if (k < steps.size()) {
				idxpos = steps[k];
			}

			std::cout << m_nRule.Board() << std::endl;
			std::cout << value << std::endl;
			std::cout << "-----" << std::endl;

			m_nRule.Turn(m_nRole, idxpos);
			gameresult = m_nRule.Check(idxpos);

			if (gameresult != Tic::GameType::UNOVER) {
				break;
			}

			if (m_nRole == Tic::RoleType::CROSS) {
				m_nRole = Tic::RoleType::CIRCLE;
			}
			else {
				m_nRole = Tic::RoleType::CROSS;
			}
		}
	}
}
