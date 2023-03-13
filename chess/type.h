#pragma once

#include "cuda.h"
#include "../Eigen/Core"

namespace util
{
	// element size
	#define EZ	sizeof(float)

	// CPU矩阵 内存中列为主序 和 CUBLAS相同
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> HOSTMatrix;

	// 定义CUDA中用到的矩阵结构
	struct CUDAMatrix {
		union {
			size_t		width = 0;		// 宽度
			size_t		cols;			// 列
		};

		union {
			size_t		height;			// 高度
			size_t		rows;			// 行
		};
		
		size_t			size;			// 全部字节大小
		CUdeviceptr		data;			// 数据
	};

	CUDAMatrix CreateCUDAMatrix(const HOSTMatrix& hostm);
	CUDAMatrix CreateCUDAMatrix(size_t rows, size_t cols);
	HOSTMatrix CreateHOSTMatrix(const CUDAMatrix& cudam);
	void DestroyCUDAMatrix(CUDAMatrix& cudam);
	void CopyHostToCUDA(const HOSTMatrix& hostm, CUDAMatrix& cudam);
	void CopyCUDAToHost(const CUDAMatrix& cudam, HOSTMatrix& hostm);

}