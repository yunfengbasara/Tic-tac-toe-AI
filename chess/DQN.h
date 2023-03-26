#pragma once
#include <array>
#include <vector>
#include <map>
#include <queue>
#include "TicRule.h"
#include "neuralex.h"
#include "../Eigen/Core"

namespace chess
{
	#define NEURALDIM	9
	#define BOARDDIM	9

	class DQN 
	{
	public:
		DQN();
		~DQN();

		// 神经网络输入
		typedef std::array<int, NEURALDIM> NEURALBOARD;
		// 真实棋盘
		typedef std::array<int, BOARDDIM> BUFFERBOARD;
		typedef Eigen::Matrix3f VALUE;

		typedef struct {
			BUFFERBOARD bufferboard;
			NEURALBOARD neuralboard;
			VALUE value;
		}REPLAY, *PREPLAY;

		typedef struct {
			PREPLAY pReplay;
			int idxpos;
			float reward;
		}QITEM;

		void Create();
		void Print();

	private:
		size_t BufferSize();
		void Generate(float explore);
		void Shuffle();
		void ClearBuffer();
		void Train();
		void UpdateQTable(QITEM& item, float score);

		void GetReplay(
			const BUFFERBOARD& buffer,
			const NEURALBOARD& neural,
			PREPLAY& pReplay,
			bool bByNeural = false);
		void GetValueByNeural(const NEURALBOARD& board, VALUE& value);
		void GetValueByBuffer(const BUFFERBOARD& board, VALUE& value);

	private:
		chess::Tic m_nRule;

		std::map<BUFFERBOARD, PREPLAY> m_mReplayIndex;
		std::queue<PREPLAY> m_vReplayPool;
		std::vector<PREPLAY> m_vReplayStore;

		util::NeuralEx m_nNeural;
	};
}