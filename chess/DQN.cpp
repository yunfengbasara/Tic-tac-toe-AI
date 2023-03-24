#include "DQN.h"
#include <random>
#include <iostream>

using namespace chess;
using namespace util;
using namespace Eigen;

DQN::DQN()
{
	srand((unsigned int)time(0));
	m_nNeural.SetCostType(NeuralEx::CrossEntropy);
	m_nNeural.InitBuild({ 9, 100, 9 });
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
	float explore = 0.37;
	int times = 50000;
	for (int i = 0; i < times; i++) {
		Generate(explore);
		
		int sz = BufferSize();
		if (sz >= 1000 && sz < 2000) {
			explore = 0.27;
		}
		else if (sz >= 2000 && sz < 3000) {
			explore = 0.20;
		}
		else if (sz >= 3000) {
			explore = 0.15;
		}
	}
	Train();

	std::cout << "size:" << BufferSize() << std::endl;

	// 输入接近，导致无法区分
	//Matrix3i board1;
	//board1 << 1, 2, 1,
	//	 0, 0, 0,
	//	 1, 2, 2;

	//PREPLAY pReplay1 = new REPLAY();
	//memcpy(&pReplay1->board[0], board1.data(), m_nRule.BDSZ());
	//pReplay1->value << 
	//	0.0, 0.0, 0.0,
	//	0.0, 0.9, 0.0,
	//	0.0, 0.0, 0.0;

	//m_vReplayStore.push_back(pReplay1);
	//m_mReplayIndex.insert({ pReplay1->board, pReplay1 });

	//Matrix3i board2;
	//board2 << 0, 0, 0,
	//	0, 0, 0,
	//	0, 0, 0;

	//PREPLAY pReplay2 = new REPLAY();
	//memcpy(&pReplay2->board[0], board2.data(), m_nRule.BDSZ());
	//pReplay2->value <<
	//	0.9, 0.9, 0.9,
	//	0.9, 0.0, 0.9,
	//	0.9, 0.9, 0.9;

	//m_vReplayStore.push_back(pReplay2);
	//m_mReplayIndex.insert({ pReplay2->board, pReplay2 });

	//std::cout << BufferSize() << std::endl;

	//Train();

	//VALUE neuralval;
	//GetValueByNeural(pReplay1->board, neuralval);
	//std::cout << neuralval << std::endl;
	//GetValueByNeural(pReplay2->board, neuralval);
	//std::cout << neuralval << std::endl;;
}

void DQN::Print()
{
	// 测试的开始几步
	std::vector<int> steps = { 0,8 };

	m_nRule.Reset();

	BOARD board{};
	Tic::GameType res = Tic::UNOVER;
	Tic::RoleType role = Tic::CROSS;

	for (int i = 0; res == Tic::UNOVER; i++) {
		const auto& eibd = role == Tic::CROSS ?
			m_nRule.Board() : m_nRule.RBoard();
		memcpy(&board[0], eibd.data(), m_nRule.BDSZ());

		VALUE neuralval;
		GetValueByNeural(board, neuralval);

		PREPLAY pReplay = NULL;
		GetReplay(board, pReplay);

		// print
		std::cout << m_nRule.Board() << std::endl;
		std::cout << "board value" << std::endl;
		std::cout << pReplay->value << std::endl;

		std::cout << "neural value" << std::endl;
		std::cout << neuralval << std::endl;
		std::cout << "-----" << std::endl;

		float score;
		int row, col, pos;
		score = m_nRule.GetMaxScore(pReplay->value, row, col);
		pos = row * 3 + col;

		if (i < steps.size()) {
			pos = steps[i];
		}

		m_nRule.Turn(role, pos);
		res = m_nRule.Check(pos);

		role = role == Tic::CROSS ?
			Tic::CIRCLE : Tic::CROSS;
	}
}

size_t DQN::BufferSize()
{
	return m_mReplayIndex.size();
}

void DQN::Generate(float explore)
{
	m_nRule.Reset();

	// 用于记录上一次Q表的状态
	QITEM cross{ NULL, -1, 0.0f };
	QITEM circle{ NULL, -1, 0.0f };

	BOARD board{};
	PREPLAY pReplay = NULL;

	Tic::GameType res = Tic::UNOVER;
	Tic::RoleType role = Tic::CROSS;

	while (res == Tic::UNOVER) {
		const auto& eibd = role == Tic::CROSS ?
			m_nRule.Board() : m_nRule.RBoard();
		memcpy(&board[0], eibd.data(), m_nRule.BDSZ());

		// 查询buffer中的Q表
		// 如果没有则通过神经网络生成新Q表
		GetReplay(board, pReplay);

		// 选择最佳位置
		float score;
		int row, col, pos;
		score = m_nRule.GetMaxScore(pReplay->value, row, col);
		pos = row * 3 + col;

		bool useRandom = (std::random_device()() % 100 / 100.0f) < explore;
		if (useRandom) {
			pos = m_nRule.RandomPos();
		}

		// 更新上一轮Q表数值
		QITEM& qItem = role == Tic::CROSS ?
			cross : circle;

		UpdateQTable(qItem, score);

		// 记录本次结果
		qItem.reward = 0.0f;
		qItem.pReplay = pReplay;
		qItem.idxpos = pos;

		// 执行
		m_nRule.Turn(role, pos);
		res = m_nRule.Check(pos);

		// 游戏结束
		if (res != Tic::UNOVER){
			break;
		}

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

	UpdateQTable(circle, foScore);
	UpdateQTable(cross, fxScore);
}

void DQN::Shuffle() 
{
	auto seed = std::default_random_engine(std::random_device()());
	std::shuffle(m_vReplayStore.begin(), m_vReplayStore.end(), seed);
}

void DQN::GetReplay(
	const BOARD& board, 
	PREPLAY& pReplay, 
	bool bByNeural)
{
	auto it = m_mReplayIndex.find(board);
	if (it != m_mReplayIndex.end()) {
		pReplay = it->second;
		return;
	}

	// 从池内创建replay
	if (!m_vReplayPool.empty()) {
		pReplay = m_vReplayPool.front();
		m_vReplayPool.pop();
	}
	// 额外创建replay
	else {
		pReplay = new REPLAY();
		m_vReplayStore.push_back(pReplay);
	}

	m_mReplayIndex.insert({ board, pReplay });
	
	pReplay->board = board;

	// 默认赋值
	if (!bByNeural) {
		pReplay->value.setZero();
	}
	// 通过神经网络赋值
	else {
		GetValueByNeural(board, pReplay->value);
	}
	return;
}

void DQN::ClearBuffer()
{
	for (auto& item : m_vReplayStore) {
		m_vReplayPool.push(item);
	}
	m_mReplayIndex.clear();
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

void chess::DQN::GetValueByNeural(const BOARD& board, VALUE& value)
{
	MatrixXf in(9, 1);
	MatrixXf out(9, 1);
	in.col(0) = Map<VectorXi>((int*)board.data(), 9).cast<float>();
	m_nNeural.CalcActive(in, out);
	value = Map<Matrix3f>((float*)out.col(0).data());
}

void DQN::Train()
{
	// 归一化
	for (const auto& pReplay : m_vReplayStore) {
		pReplay->value = (pReplay->value.array() + 1.0f) / 2.0f;
	}

	// 样本总数
	size_t count = BufferSize();
	m_nNeural.SetTotalItem(count);

	int epochs = 100;
	int batch = 32;

	MatrixXf mi(9, batch);
	MatrixXf mt(9, batch);
	MatrixXf so;
	float loss = 0;

	for (int i = 0; i < epochs; i++) {
		Shuffle();

		for (int k = 0; k < count; k += batch) {
			int sz = batch;
			if (k + batch > count) {
				sz = count - k;
			}

			if (sz == 0) {
				continue;
			}

			if (mi.cols() != sz) {
				mi.resize(NoChange, sz);
				mt.resize(NoChange, sz);
			}

			for (int m = 0; m < sz; m++) {
				const PREPLAY& pReplay = m_vReplayStore[k + m];
				mi.col(m) = Map<VectorXi>((int*)pReplay->board.data(), 9).cast<float>();
				mt.col(m) = Map<VectorXf>((float*)pReplay->value.data(), 9);
			}

			if (!m_nNeural.SetSample(mi, mt)) {
				break;
			}

			m_nNeural.SGD();
		}

		//m_nNeural.CompareSample(mi, mt, so, loss);
		//std::cout << "loss " << loss << std::endl;
	}
}