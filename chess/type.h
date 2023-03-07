#pragma once

#include "cuda.h"
#include "../Eigen/Core"

namespace util
{
	// blockthreads
	#define BLOCK_SIZE 32

	// element size
	#define EZ	sizeof(float)

	// ����CUDA�еľ���������������,��˶��尴�����ȵ�CPU����
	typedef Eigen::Matrix<
		float,
		Eigen::Dynamic,
		Eigen::Dynamic,
		Eigen::RowMajor> HOSTMatrix;

	// ����CUDA���õ��ľ���ṹ
	struct CUDAMatrix {
		size_t		height = 0;		// ��������
		size_t		width = 0;		// ��������
		size_t		stride = 0;		// һ���ֽڴ�С:width * EZ
		size_t		size = 0;		// ����ȫ���ֽڴ�С:stride * height
		size_t		pitch = 0;		// ����cuda������һ���ֽڴ�С
		size_t		pitchwidth = 0;	// ����cuda������һ��Ԫ�ظ���:pitch / EZ
		CUdeviceptr data = 0;		// ��������
	};

	CUDAMatrix CreateCUDAMatrix(const HOSTMatrix& hostm);
	CUDAMatrix CreateCUDAMatrix(int h, int w);
	CUresult DestroyCUDAMatrix(CUDAMatrix& cudam);
	CUresult CopyHostToCUDA(const HOSTMatrix& hostm, CUDAMatrix& cudam, CUstream stream = nullptr);
	CUresult CopyCUDAToHost(const CUDAMatrix& cudam, HOSTMatrix& hostm, CUstream stream = nullptr);

}