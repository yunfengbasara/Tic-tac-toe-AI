#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include "../Eigen/Core"
#include "type.h"

namespace util
{
	class Analyze_IDX {
	public:
		Analyze_IDX();
		~Analyze_IDX();

		bool SetSample(
			const std::wstring& pathin,
			const std::wstring& pathout
		);

		void GetInOut(int& in, int& out);
		void Shuffle();
		
		int ReadSample(
			Eigen::MatrixXf& matin, 
			Eigen::MatrixXf& matout,
			int batch
		);

		int ReadRandom(
			Eigen::MatrixXf& matin,
			Eigen::MatrixXf& matout,
			int batch
		);

		int ReadSample(
			HOSTMatrix& matin,
			HOSTMatrix& matout,
			int batch
		);

		int ReadRandom(
			HOSTMatrix& matin,
			HOSTMatrix& matout,
			int batch
		);

	private:
		bool InitHandle(const std::wstring& path,
			HANDLE& hFile, HANDLE& hMap, LPBYTE& lpMem);

		bool CheckFile(LPBYTE lpMem, DWORD fileSize);
		bool MatchFile(LPBYTE lpMemIn, LPBYTE lpMemOut, DWORD& items);

		bool InitRowCol(LPBYTE lpMem, DWORD& rows, DWORD& cols);

	private:
		HANDLE m_hFileIn = INVALID_HANDLE_VALUE;
		HANDLE m_hMapIn = NULL;
		LPBYTE m_lpMemIn = NULL;

		HANDLE m_hFileOut = INVALID_HANDLE_VALUE;
		HANDLE m_hMapOut = NULL;
		LPBYTE m_lpMemOut = NULL;

		DWORD m_nItems = 0;
		DWORD m_nRows = 0;
		DWORD m_nCols = 0;
		DWORD m_nOutDim = 10;

		std::vector<DWORD> m_vIndex;
		DWORD m_nStart = 0;
	};
}