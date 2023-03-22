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
	class DQN 
	{
	public:
		DQN();
		~DQN();

		typedef std::array<int, 9> BOARD;
		typedef Eigen::Matrix3f VALUE;

		typedef struct {
			BOARD board;
			VALUE value;
		}REPLAY, *PREPLAY;

		typedef struct {
			PREPLAY pReplay;
			int idxpos;
			float reward;
		}QITEM;

		void Create();

	private:
		size_t BufferSize();
		void Generate();
		void Shuffle();
		PREPLAY UpdateBuffer(const BOARD& board, const VALUE& value);
		void ClearBuffer();

		void UpdateQTable(QITEM& item, float score);

	private:
		chess::Tic m_nRule;

		int m_nMaxReplaySize;

		std::map<BOARD, PREPLAY> m_mReplayIndex;
		std::queue<PREPLAY> m_vReplayPool;
		std::vector<PREPLAY> m_vReplayStore;

		util::NeuralEx m_nNeural;
	};
}