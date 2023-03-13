#pragma once

#include "cuda.h"
#include "../Eigen/Core"

namespace util
{
	// element size
	#define EZ	sizeof(float)

	// CPU���� �ڴ�����Ϊ���� �� CUBLAS��ͬ
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> HOSTMatrix;

	// ����CUDA���õ��ľ���ṹ
	struct CUDAMatrix {
		union {
			size_t		width = 0;		// ���
			size_t		cols;			// ��
		};

		union {
			size_t		height;			// �߶�
			size_t		rows;			// ��
		};
		
		size_t			size;			// ȫ���ֽڴ�С
		CUdeviceptr		data;			// ����
	};

	CUDAMatrix CreateCUDAMatrix(const HOSTMatrix& hostm);
	CUDAMatrix CreateCUDAMatrix(size_t rows, size_t cols);
	HOSTMatrix CreateHOSTMatrix(const CUDAMatrix& cudam);
	void DestroyCUDAMatrix(CUDAMatrix& cudam);
	void CopyHostToCUDA(const HOSTMatrix& hostm, CUDAMatrix& cudam);
	void CopyCUDAToHost(const CUDAMatrix& cudam, HOSTMatrix& hostm);

}