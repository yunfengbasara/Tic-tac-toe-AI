#include "TicRule.h"
#include <random>
#include <cmath>
#include <iostream>
using namespace chess;

chess::Tic::Tic()
{
	Reset();
}

chess::Tic::~Tic()
{
}

void chess::Tic::Reset()
{
	m_nEmpCnt = 9;

	// 低九位为可用位置
	m_nIndex = 0x01FF;

	m_nBoard.setZero();

	m_nRBoard.setZero();
}

const Eigen::Matrix3i& chess::Tic::Board()
{
	return m_nBoard;
}

bool chess::Tic::Create(const std::vector<int>& steps, 
	Eigen::Matrix3i& board, GameType& type, int& lp)
{
	RoleType role = CROSS;

	for (const auto& pos : steps) {
		if (!Turn(role, pos)) {
			return false;
		}

		type = Check(pos);
		if (type != UNOVER) {
			board = m_nBoard;
			lp = pos;
			return true;
		}

		role = role == CROSS ? CIRCLE : CROSS;
	}

	return false;
}

uint16_t chess::Tic::RandomPos()
{
	int cnt = std::random_device()() % m_nEmpCnt;
	uint16_t pidx = m_nIndex;
	while (cnt > 0) {
		pidx = pidx & (pidx - 1);
		cnt--;
	}

	uint16_t t = pidx & (pidx - 1);
	return std::log2(t ^ pidx);
}

uint16_t chess::Tic::MaxScorePos(const Eigen::Matrix3f& score)
{
	uint16_t midx = 100;

	uint16_t pidx = m_nIndex;
	while (pidx > 0) {
		uint16_t t = pidx & (pidx - 1);
		uint16_t idx = std::log2(t ^ pidx);
		pidx = t;

		if (midx == 100) {
			midx = idx;
			continue;
		}

		if (score(midx / 3, midx % 3) <
			score(idx / 3, idx % 3)) {
			midx = idx;
		}
	}

	return midx;
}

Eigen::Matrix3f chess::Tic::CreateValue(float score)
{
	Eigen::Matrix3f value = Eigen::Matrix3f::Zero();

	uint16_t pidx = m_nIndex;
	while (pidx > 0) {
		uint16_t t = pidx & (pidx - 1);
		uint16_t idx = std::log2(t ^ pidx);
		pidx = t;

		value(idx / 3, idx % 3) = score;
	}
	return value;
}

bool chess::Tic::Turn(RoleType role, int idx)
{
	uint16_t pos = 1 << idx;
	if ((m_nIndex & pos) == 0) {
		return false;
	}

	m_nEmpCnt--;

	m_nIndex &= ~pos;

	int n = role == CROSS ? 1 : 2;
	m_nBoard(idx / 3, idx % 3) = n;

	int rn = role != CROSS ? 1 : 2;
	m_nRBoard(idx / 3, idx % 3) = rn;

	return true;
}

bool chess::Tic::Revoke(int idx)
{
	uint16_t pos = 1 << idx;
	if ((m_nIndex & pos) == 1) {
		return false;
	}

	m_nEmpCnt++;

	m_nIndex |= pos;

	m_nBoard(idx / 3, idx % 3) = 0;

	m_nRBoard(idx / 3, idx % 3) = 0;

	return true;
}

Tic::GameType chess::Tic::Check(int idx)
{
	int x = idx / 3;
	int y = idx % 3;

	int r = m_nBoard(x, y);
	if (r == 0) {
		return UNOVER;
	}

	// 上下 左右 左上右下 右上左下
	int dx[8] = { -1, 1, 0, 0, -1, 1, -1, 1 };
	int dy[8] = { 0, 0, -1, 1, -1, 1, 1, -1 };
	for (int i = 0; i < 8; i += 2) {
		int count = 1;

		int tx = x + dx[i];
		int ty = y + dy[i];
		while (tx >= 0 && tx < 3 
			&& ty >= 0 && ty < 3) {
			if (r != m_nBoard(tx, ty)) {
				break;
			}

			tx += dx[i];
			ty += dy[i];
			count++;
		}

		tx = x + dx[i + 1];
		ty = y + dy[i + 1];
		while (tx >= 0 && tx < 3
			&& ty >= 0 && ty < 3) {
			if (r != m_nBoard(tx, ty)) {
				break;
			}

			tx += dx[i + 1];
			ty += dy[i + 1];
			count++;
		}

		if (count == 3) {
			return r == 1 ? CROSSW : CIRCLEW;
		}
	}

	// 没有剩余空间
	if (m_nIndex == 0) {
		return DRAW;
	}

	return UNOVER;
}

void chess::Tic::Reverse()
{
	std::swap(m_nBoard, m_nRBoard);
}
