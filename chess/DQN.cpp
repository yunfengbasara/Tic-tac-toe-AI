#include "DQN.h"
#include <random>
#include <iostream>

using namespace chess;
using namespace util;
using namespace Eigen;

DQN::DQN()
{
	m_nMaxReplaySize = 1;

	srand((unsigned int)time(0));
	m_nNeural.SetCostType(NeuralEx::CrossEntropy);
	m_nNeural.InitBuild({ 9, 800, 9 });
	m_nNeural.SetLearnRate(0.1);
	m_nNeural.SetRegularization(5.0);
}

DQN::~DQN() 
{
	for (auto& item : m_vReplayStore) {
		delete item;
	}
}

void DQN::Create() 
{
	int times = 1;
	for (int i = 0; i < times; i++) {
		Generate();

		if (BufferSize() >= m_nMaxReplaySize) {
			break;
		}
	}

	// 训练设置
	//Shuffle();
	// buffer size可能会略有变化
	m_nNeural.SetTotalItem(BufferSize());

	for (auto& item : m_vReplayStore) {
		Matrix3i board(item->board.data());
		std::cout << board << std::endl;
		Matrix3f value(item->value.data());
		std::cout << value << std::endl;
		std::cout << "-----" << std::endl;
	}
}

size_t DQN::BufferSize()
{
	return m_mReplayIndex.size();
}

void DQN::Generate()
{
	m_nRule.Reset();

	// 用于记录上一次Q表的状态
	QITEM cross{ NULL, -1, 0.0f };
	QITEM circle{ NULL, -1, 0.0f };

	// 通过神经网络生成的输出
	MatrixXf in(9, 1);
	MatrixXf out(9, 1);

	BOARD board{};
	VALUE value{};
	Tic::GameType res = Tic::UNOVER;
	Tic::RoleType role = Tic::CROSS;

	while (res == Tic::UNOVER) {
		QITEM& qItem = role == Tic::CROSS ?
			cross : circle;

		const auto& eibd = role == Tic::CROSS ?
			m_nRule.Board() : m_nRule.RBoard();
		memcpy(&board[0], eibd.data(), m_nRule.BDSZ());

		PREPLAY pReplay = NULL;

		const auto& it = m_mReplayIndex.find(board);
		// 查询buffer中的Q表
		if (it != m_mReplayIndex.end()) {
			value = it->second->value;
			pReplay = it->second;
		}
		// 通过神经网络生成新Q表
		else {
			in.col(0) = Map<VectorXi>((int*)eibd.data(), 9).cast<float>();
			m_nNeural.CalcActive(in, out);
			value = Map<Matrix3f>((float*)out.col(0).data());
			m_nRule.SetEmptyOnRole(value);
			pReplay = UpdateBuffer(board, value);
		}

		// 选择最佳位置
		float score;
		int row, col, pos;
		score = m_nRule.GetMaxScore(value, row, col);
		pos = row * 3 + col;

		// 更新上一轮Q表数值
		UpdateQTable(qItem, score);

		// 记录本次结果
		qItem.reward = 0.0f;
		qItem.pReplay = pReplay;
		qItem.idxpos = pos;

		m_nRule.Turn(role, pos);
		res = m_nRule.Check(pos);

		// 交换角色
		role = role == Tic::CROSS ?
			Tic::CIRCLE : Tic::CROSS;
	}

	// 更新最后一轮结果
	float foScore = 0.0f;
	float fxScore = 0.0f;

	if (res == Tic::DRAW) {
		foScore = -0.01f;
		fxScore = -0.01f;
	}
	else if (role == Tic::CROSS) {
		foScore = -1.0f;
		fxScore = 1.0f;
	}
	else {
		foScore = 1.0f;
		fxScore = -1.0f;
	}

	UpdateQTable(cross, foScore);
	UpdateQTable(circle, fxScore);
}

void DQN::Shuffle() 
{
	auto seed = std::default_random_engine(std::random_device()());
	std::shuffle(m_vReplayStore.begin(), m_vReplayStore.end(), seed);
}

DQN::PREPLAY DQN::UpdateBuffer(const BOARD& board, const VALUE& value)
{
	auto it = m_mReplayIndex.find(board);
	if (it != m_mReplayIndex.end()) {
		PREPLAY pReplay = it->second;
		pReplay->value = value;
		return pReplay;
	}

	// 池内有空间
	if (!m_vReplayPool.empty()) {
		PREPLAY pReplay = m_vReplayPool.front();
		pReplay->board = board;
		pReplay->value = value;
		m_vReplayPool.pop();
		m_mReplayIndex.insert(make_pair(board, pReplay));
		return pReplay;
	}

	// 空间不足
	PREPLAY pReplay = new REPLAY();
	pReplay->board = board;
	pReplay->value = value;
	m_vReplayStore.push_back(pReplay);
	m_mReplayIndex.insert(make_pair(board, pReplay));
	return pReplay;
}

void DQN::ClearBuffer()
{
	for (auto& item : m_vReplayStore) {
		m_vReplayPool.push(item);
		m_mReplayIndex.erase(item->board);
	}
}

void DQN::UpdateQTable(QITEM& item, float score)
{
	if (item.idxpos == -1) {
		return;
	}

	float eta = 0.17;
	auto& v = item.pReplay->value(item.idxpos / 3, item.idxpos % 3);
	v = (1 - eta) * v + eta * (item.reward + 0.8 * score);
}
