#pragma once

#include "cuda.h"
#include "../Eigen/Core"

namespace util
{
	// blockthreads
	#define BLOCK_SIZE 32

	// element size
	#define EZ	sizeof(float)

	// 由于CUDA中的矩阵按照行优先排列,因此定义按行优先的CPU矩阵
	typedef Eigen::Matrix<
		float,
		Eigen::Dynamic,
		Eigen::Dynamic,
		Eigen::RowMajor> HOSTMatrix;

	// 定义CUDA中用到的矩阵结构
	struct CUDAMatrix {
		size_t		height = 0;		// 矩阵行数
		size_t		width = 0;		// 矩阵列数
		size_t		stride = 0;		// 一行字节大小:width * EZ
		size_t		size = 0;		// 矩阵全部字节大小:stride * height
		size_t		pitch = 0;		// 经过cuda对齐后的一行字节大小
		size_t		pitchwidth = 0;	// 经过cuda对齐后的一行元素个数:pitch / EZ
		CUdeviceptr data = 0;		// 矩阵数据
	};

	CUDAMatrix CreateCUDAMatrix(const HOSTMatrix& hostm);
	CUDAMatrix CreateCUDAMatrix(int h, int w);
	CUresult DestroyCUDAMatrix(CUDAMatrix& cudam);
	CUresult CopyHostToCUDA(const HOSTMatrix& hostm, CUDAMatrix& cudam, CUstream stream = nullptr);
	CUresult CopyCUDAToHost(const CUDAMatrix& cudam, HOSTMatrix& hostm, CUstream stream = nullptr);

}