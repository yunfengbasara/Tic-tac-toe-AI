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

	// ѵ������
	//Shuffle();
	// buffer size���ܻ����б仯
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

	// ���ڼ�¼��һ��Q���״̬
	QITEM cross{ NULL, -1, 0.0f };
	QITEM circle{ NULL, -1, 0.0f };

	// ͨ�����������ɵ����
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
		// ��ѯbuffer�е�Q��
		if (it != m_mReplayIndex.end()) {
			value = it->second->value;
			pReplay = it->second;
		}
		// ͨ��������������Q��
		else {
			in.col(0) = Map<VectorXi>((int*)eibd.data(), 9).cast<float>();
			m_nNeural.CalcActive(in, out);
			value = Map<Matrix3f>((float*)out.col(0).data());
			m_nRule.SetEmptyOnRole(value);
			pReplay = UpdateBuffer(board, value);
		}

		// ѡ�����λ��
		float score;
		int row, col, pos;
		score = m_nRule.GetMaxScore(value, row, col);
		pos = row * 3 + col;

		// ������һ��Q����ֵ
		UpdateQTable(qItem, score);

		// ��¼���ν��
		qItem.reward = 0.0f;
		qItem.pReplay = pReplay;
		qItem.idxpos = pos;

		m_nRule.Turn(role, pos);
		res = m_nRule.Check(pos);

		// ������ɫ
		role = role == Tic::CROSS ?
			Tic::CIRCLE : Tic::CROSS;
	}

	// �������һ�ֽ��
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

	// �����пռ�
	if (!m_vReplayPool.empty()) {
		PREPLAY pReplay = m_vReplayPool.front();
		pReplay->board = board;
		pReplay->value = value;
		m_vReplayPool.pop();
		m_mReplayIndex.insert(make_pair(board, pReplay));
		return pReplay;
	}

	// �ռ䲻��
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
