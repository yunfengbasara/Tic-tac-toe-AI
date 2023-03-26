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
	m_nNeural.InitBuild({ NEURALDIM, 160, BOARDDIM });
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
			explore = 0.31;
		}
		else if (sz >= 2000 && sz < 3000) {
			explore = 0.27;
		}
		else if (sz >= 3000) {
			explore = 0.20;
		}
	}
	Train();

	std::cout << "size:" << BufferSize() << std::endl;

	// 两组数据无法拟合，这就离谱
	//PREPLAY pReplay1 = new REPLAY();
	//pReplay1->neuralboard.fill(0);
	//for (int i = 0; i < 9; i++) {
	//	pReplay1->neuralboard[i] = 100;
	//}
	//pReplay1->bufferboard.fill(0);

	//pReplay1->value.setZero();
	//pReplay1->value(0, 0) = 0.34;
	//pReplay1->value(1, 1) = 0.78;

	//m_vReplayStore.push_back(pReplay1);
	//m_mReplayIndex.insert({ pReplay1->bufferboard, pReplay1 });

	//PREPLAY pReplay2 = new REPLAY();
	//pReplay2->neuralboard.fill(0);
	//for (int i = 0; i < 9; i++) {
	//	pReplay2->neuralboard[i] = i;
	//}
	//pReplay2->bufferboard.fill(1);

	//pReplay2->value.setConstant(1);
	//pReplay2->value(0, 0) = 0.11;
	//pReplay2->value(1, 1) = 0.56;

	//m_vReplayStore.push_back(pReplay2);
	//m_mReplayIndex.insert({ pReplay2->bufferboard, pReplay2 });

	//std::cout << BufferSize() << std::endl;

	//Train();

	//VALUE neuralval;
	//GetValueByNeural(pReplay1->neuralboard, neuralval);
	//std::cout << neuralval << std::endl;
	//GetValueByNeural(pReplay2->neuralboard, neuralval);
	//std::cout << neuralval << std::endl;
}

void DQN::Print()
{
	// 测试的开始几步
	std::vector<int> steps = {0,8 };

	m_nRule.Reset();

	BUFFERBOARD buffer{};
	NEURALBOARD neural{};
	VALUE bufferval;
	VALUE neuralval;

	Tic::GameType res = Tic::UNOVER;
	Tic::RoleType role = Tic::CROSS;

	for (int i = 0; res == Tic::UNOVER; i++) {
		const auto& eibd = role == Tic::CROSS ?
			m_nRule.Board() : m_nRule.RBoard();
		memcpy(&buffer[0], eibd.data(), m_nRule.BDSZ());

		const auto& einbd = role == Tic::CROSS ?
			m_nRule.NeuralBoard() : m_nRule.RNeuralBoard();
		memcpy(&neural[0], einbd.data(), m_nRule.NBDSZ());

		
		GetValueByNeural(neural, neuralval);
		GetValueByBuffer(buffer, bufferval);

		// print
		std::cout << m_nRule.Board() << std::endl;
		std::cout << "table value" << std::endl;
		std::cout << bufferval << std::endl;

		std::cout << "neural value" << std::endl;
		std::cout << neuralval << std::endl;
		std::cout << "-----" << std::endl;

		// 采用神经网络的策略下棋
		int row, col, pos;
		m_nRule.GetMaxScore(neuralval, row, col);
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

	BUFFERBOARD buffer{};
	NEURALBOARD neural{};
	PREPLAY pReplay = NULL;

	Tic::GameType res = Tic::UNOVER;
	Tic::RoleType role = Tic::CROSS;

	while (res == Tic::UNOVER) {
		const auto& eibd = role == Tic::CROSS ?
			m_nRule.Board() : m_nRule.RBoard();
		memcpy(&buffer[0], eibd.data(), m_nRule.BDSZ());

		const auto& einbd = role == Tic::CROSS ?
			m_nRule.NeuralBoard() : m_nRule.RNeuralBoard();
		memcpy(&neural[0], einbd.data(), m_nRule.NBDSZ());

		// 查询buffer中的Q表
		// 如果没有则通过神经网络生成新Q表
		GetReplay(buffer, neural, pReplay);

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
	const BUFFERBOARD& buffer,
	const NEURALBOARD& neural,
	PREPLAY& pReplay, 
	bool bByNeural)
{
	auto it = m_mReplayIndex.find(buffer);
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

	m_mReplayIndex.insert({ buffer, pReplay });
	
	pReplay->bufferboard = buffer;
	pReplay->neuralboard = neural;

	// 默认赋值
	if (!bByNeural) {
		GetValueByBuffer(buffer, pReplay->value);
	}
	// 通过神经网络赋值
	else {
		GetValueByNeural(neural, pReplay->value);
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

void chess::DQN::GetValueByNeural(const NEURALBOARD& board, VALUE& value)
{
	MatrixXf in(board.size(), 1);
	MatrixXf out(value.size(), 1);
	in.col(0) = Map<VectorXi>((int*)board.data(), board.size()).cast<float>();
	m_nNeural.CalcActive(in, out);
	value = Map<Matrix3f>((float*)out.col(0).data());
	m_nRule.SetEmptyOnRole(value);
}

void chess::DQN::GetValueByBuffer(const BUFFERBOARD& board, VALUE& value)
{
	auto it = m_mReplayIndex.find(board);
	if (it != m_mReplayIndex.end()) {
		value = it->second->value;
		return;
	}

	// 采用全0初始化
	//value.setZero();

	// 采用随机初始值Q表
	value.setRandom();
	// 将走过的位置置0
	m_nRule.SetEmptyOnRole(value);
}

void DQN::Train()
{
	// 归一化
	for (const auto& pReplay : m_vReplayStore) {
		//// 0无需归一化处理策略
		//for (int i = 0; i < pReplay->value.array().size(); i++) {
		//	float val = pReplay->value.array()(i);
		//	if (val > 0.000001 || val < -0.000001) {
		//		pReplay->value.array()(i) = (val + 1.0f) / 2.0f;
		//	}
		//}
		// 只保留最大值策略
		int k = 0;
		float f = 0;
		for (int i = 0; i < pReplay->value.array().size(); i++) {
			float val = pReplay->value.array()(i);
			if (val < 0.000001 && val > -0.000001) {
				continue;
			}

			// 归一化处理
			val = (val + 1.0f) / 2.0f;
			if (val > f) {
				f = val;
				k = i;
			}
			pReplay->value.array()(i) = 0;
		}
		pReplay->value.array()(k) = f;
	}

	// 样本总数
	size_t count = BufferSize();
	m_nNeural.SetTotalItem(count);

	int epochs = 100;
	int batch = 32;

	MatrixXf mi(NEURALDIM, batch);
	MatrixXf mt(BOARDDIM, batch);
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

				mi.col(m) = Map<VectorXi>((int*)pReplay->neuralboard.data(), 
					pReplay->neuralboard.size()).cast<float>();

				mt.col(m) = Map<VectorXf>((float*)pReplay->value.data(), 
					pReplay->value.size());
			}

			if (!m_nNeural.SetSample(mi, mt)) {
				break;
			}

			m_nNeural.SGD();
		}

		//m_nNeural.CompareSample(mi, mt, so, loss);
		//std::cout << "loss " << loss << " epoch:" << i << std::endl;
	}
}