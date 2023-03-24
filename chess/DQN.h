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
		void Print();

	private:
		size_t BufferSize();
		void Generate(float explore);
		void Shuffle();
		void GetReplay(
			const BOARD& board, 
			PREPLAY& pReplay,
			bool bByNeural = false);
		void ClearBuffer();
		void Train();

		void UpdateQTable(QITEM& item, float score);
		void GetValueByNeural(const BOARD& board, VALUE& value);

	private:
		chess::Tic m_nRule;

		std::map<BOARD, PREPLAY> m_mReplayIndex;
		std::queue<PREPLAY> m_vReplayPool;
		std::vector<PREPLAY> m_vReplayStore;

		util::NeuralEx m_nNeural;
	};
}