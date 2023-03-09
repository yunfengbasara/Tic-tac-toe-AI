#include "Analyze.h"
#include "util.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <winsock.h>
#pragma comment(lib, "Ws2_32.lib")

using namespace util;
using namespace Eigen;

util::Analyze_IDX::Analyze_IDX() 
{
}

util::Analyze_IDX::~Analyze_IDX()
{
	if (m_lpMemIn != NULL) {
		::UnmapViewOfFile(m_lpMemIn);
	}
	if (m_hMapIn != NULL) {
		::CloseHandle(m_hMapIn);
	}
	if (m_hFileIn != INVALID_HANDLE_VALUE) {
		::CloseHandle(m_hFileIn);
	}

	if (m_lpMemOut != NULL) {
		::UnmapViewOfFile(m_lpMemOut);
	}
	if (m_hMapOut != NULL) {
		::CloseHandle(m_hMapOut);
	}
	if (m_hFileOut != INVALID_HANDLE_VALUE) {
		::CloseHandle(m_hFileOut);
	}
}

bool util::Analyze_IDX::SetSample(
	const std::wstring& pathin,
	const std::wstring& pathout) 
{
	if (!InitHandle(pathin, m_hFileIn, m_hMapIn, m_lpMemIn)) {
		return false;
	}

	if (!InitHandle(pathout, m_hFileOut, m_hMapOut, m_lpMemOut)) {
		return false;
	}

	DWORD fileSizeIn = 0;
	fileSizeIn = ::GetFileSize(m_hFileIn, NULL);
	if (!CheckFile(m_lpMemIn, fileSizeIn)) {
		return false;
	}

	DWORD fileSizeOut = 0;
	fileSizeOut = ::GetFileSize(m_hFileOut, NULL);
	if (!CheckFile(m_lpMemOut, fileSizeOut)) {
		return false;
	}

	if (!MatchFile(m_lpMemIn, m_lpMemOut, m_nItems)) {
		return false;
	}

	if (!InitRowCol(m_lpMemIn, m_nRows, m_nCols)) {
		return false;
	}

	m_vIndex.resize(m_nItems);
	std::generate(
		m_vIndex.begin(),
		m_vIndex.end(),
		[idx = 0]() mutable{
		return idx++;
	});

	return true;
}

void util::Analyze_IDX::GetInOut(int& in, int& out)
{
	// 输入28 * 28灰度图像素
	in = m_nRows * m_nCols;

	// 输出0-9结果,用10维输出代表
	out = m_nOutDim;
}

void util::Analyze_IDX::Shuffle()
{
	m_nStart = 0;

	auto seed = std::default_random_engine(std::random_device()());
	std::shuffle(m_vIndex.begin(), m_vIndex.end(), seed);
}

int util::Analyze_IDX::ReadSample(
	MatrixXf& matin,
	MatrixXf& matout,
	int batch)
{
	if (m_nStart + batch > m_nItems) {
		batch = m_nItems - m_nStart;
	}

	if (batch != matin.cols()) {
		matin.resize(NoChange, batch);
		matout.resize(NoChange, batch);
	}

	// 输出清零
	// 输出10个维度,在对应编号位设置1
	matout.setZero();

	typedef Vector<BYTE, Eigen::Dynamic> VectorXb;

	for (int i = 0; i < batch; i++) {
		DWORD idx = m_vIndex[m_nStart++];

		DWORD sepin = idx * m_nRows * m_nCols;
		LPBYTE lpin = m_lpMemIn + sizeof(DWORD) * 4 + sepin;
		matin.col(i) = Map<VectorXb>(lpin, m_nRows * m_nCols).cast<float>();

		DWORD sepout = idx * 1;
		LPBYTE lpout = m_lpMemOut + sizeof(DWORD) * 2 + sepout;
		BYTE sign = *lpout;
		matout.col(i)(sign) = 1.0f;
	}
	
	return batch;
}

int util::Analyze_IDX::ReadRandom(
	Eigen::MatrixXf& matin,
	Eigen::MatrixXf& matout,
	int batch) 
{
	std::vector<DWORD> idxs(batch);
	for (int i = 0; i < batch; i++) {
		idxs[i] = std::random_device()() % m_nItems;
	}

	if (batch != matin.cols()) {
		matin.resize(NoChange, batch);
		matout.resize(NoChange, batch);
	}

	matout.setZero();

	typedef Vector<BYTE, Eigen::Dynamic> VectorXb;

	for (int i = 0; i < batch; i++) {
		DWORD sepin = idxs[i] * m_nRows * m_nCols;
		LPBYTE lpin = m_lpMemIn + sizeof(DWORD) * 4 + sepin;
		matin.col(i) = Map<VectorXb>(lpin, m_nRows * m_nCols).cast<float>();

		DWORD sepout = idxs[i] * 1;
		LPBYTE lpout = m_lpMemOut + sizeof(DWORD) * 2 + sepout;
		BYTE sign = *lpout;
		matout.col(i)(sign) = 1.0f;
	}

	return batch;
}

int util::Analyze_IDX::ReadSample(
	HOSTMatrix& matin, 
	HOSTMatrix& matout, 
	int batch)
{
	if (m_nStart + batch > m_nItems) {
		batch = m_nItems - m_nStart;
	}

	if (batch != matin.cols()) {
		matin.resize(NoChange, batch);
		matout.resize(NoChange, batch);
	}

	// 输出清零
	// 输出10个维度,在对应编号位设置1
	matout.setZero();

	typedef Vector<BYTE, Eigen::Dynamic> VectorXb;

	for (int i = 0; i < batch; i++) {
		DWORD idx = m_vIndex[m_nStart++];

		DWORD sepin = idx * m_nRows * m_nCols;
		LPBYTE lpin = m_lpMemIn + sizeof(DWORD) * 4 + sepin;
		matin.col(i) = Map<VectorXb>(lpin, m_nRows * m_nCols).cast<float>();

		DWORD sepout = idx * 1;
		LPBYTE lpout = m_lpMemOut + sizeof(DWORD) * 2 + sepout;
		BYTE sign = *lpout;
		matout.col(i)(sign) = 1.0f;
	}

	return batch;
}

int util::Analyze_IDX::ReadRandom(
	HOSTMatrix& matin, 
	HOSTMatrix& matout, 
	int batch)
{
	std::vector<DWORD> idxs(batch);
	for (int i = 0; i < batch; i++) {
		idxs[i] = std::random_device()() % m_nItems;
	}

	if (batch != matin.cols()) {
		matin.resize(NoChange, batch);
		matout.resize(NoChange, batch);
	}

	matout.setZero();

	typedef Vector<BYTE, Eigen::Dynamic> VectorXb;

	for (int i = 0; i < batch; i++) {
		DWORD sepin = idxs[i] * m_nRows * m_nCols;
		LPBYTE lpin = m_lpMemIn + sizeof(DWORD) * 4 + sepin;
		matin.col(i) = Map<VectorXb>(lpin, m_nRows * m_nCols).cast<float>();

		DWORD sepout = idxs[i] * 1;
		LPBYTE lpout = m_lpMemOut + sizeof(DWORD) * 2 + sepout;
		BYTE sign = *lpout;
		matout.col(i)(sign) = 1.0f;
	}

	return batch;
}

bool util::Analyze_IDX::InitHandle(const std::wstring& path,
	HANDLE& hFile, HANDLE& hMap, LPBYTE& lpMem)
{
	hFile = CreateFile(path.c_str(), GENERIC_READ, 0,
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) {
		return false;
	}

	hMap = ::CreateFileMapping(hFile,
		NULL, PAGE_READONLY, 0, 0, NULL);
	if (hMap == NULL) {
		return false;
	}

	lpMem = (LPBYTE)::MapViewOfFile(hMap,
		FILE_MAP_READ, 0, 0, 0);
	if (lpMem == NULL) {
		return false;
	}

	return true;
}

bool util::Analyze_IDX::CheckFile(LPBYTE lpMem, DWORD fileSize)
{
	// 开始读文件位置
	DWORD fileStart = 0;

	// idx head
	DWORD magic = 0x00000800;
	if (fileStart + sizeof(magic) > fileSize) {
		return false;
	}

	DWORD head = *(DWORD*)(lpMem + fileStart);
	head = ntohl(head);
	if ((head & 0xFFFFFFF0) != magic) {
		return false;
	}

	DWORD dimension = head & 0x0000000F;
	if (dimension != 1 && dimension != 3) {
		return false;
	}

	fileStart += sizeof(magic);

	//  number of items
	DWORD items = 0;
	if (fileStart + sizeof(items) > fileSize) {
		return false;
	}

	items = *(DWORD*)(lpMem + fileStart);
	items = ntohl(items);

	fileStart += sizeof(items);

	if (dimension == 1) {
		fileStart += items;
		return fileStart == fileSize;
	}

	DWORD rows = 0, cols = 0;
	if (fileStart + sizeof(DWORD) * 2 > fileSize) {
		return false;
	}

	rows = *(DWORD*)(lpMem + fileStart);
	rows = ntohl(rows);
	cols = *(DWORD*)(lpMem + fileStart + sizeof(DWORD));
	cols = ntohl(cols);

	fileStart += sizeof(DWORD) * 2;
	fileStart += items * rows * cols;

	return fileStart == fileSize;
}

bool util::Analyze_IDX::MatchFile(LPBYTE lpMemIn, LPBYTE lpMemOut, DWORD& items)
{
	DWORD itemsin = *(DWORD*)(lpMemIn + sizeof(DWORD));
	itemsin = ntohl(itemsin);

	DWORD itemsout = *(DWORD*)(lpMemOut + sizeof(DWORD));
	itemsout = ntohl(itemsout);

	if (itemsin != itemsout) {
		return false;
	}

	m_nItems = itemsin;
	return true;
}

bool util::Analyze_IDX::InitRowCol(LPBYTE lpMem, DWORD& rows, DWORD& cols)
{
	DWORD head = *(DWORD*)(lpMem + 0);
	head = ntohl(head);

	DWORD dimension = head & 0x0000000F;
	if (dimension != 3) {
		return false;
	}

	rows = *(DWORD*)(lpMem + sizeof(DWORD) * 2);
	rows = ntohl(rows);

	cols = *(DWORD*)(lpMem + sizeof(DWORD) * 3);
	cols = ntohl(cols);

	return true;
}