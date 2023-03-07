#pragma once

#include "cuda.h"
#include "../Eigen/Core"

namespace util
{
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
		size_t		pitchcols = 0;	// ����cuda������һ��Ԫ�ظ���:pitch / EZ
		CUdeviceptr data = 0;		// ��������
	};

	CUresult CreateCUDAMatrix(const HOSTMatrix& hostm, CUDAMatrix& cudam);
	CUresult DestroyCUDAMatrix(CUDAMatrix& cudam);
	CUresult CopyHostToCUDA(HOSTMatrix& hostm, CUDAMatrix& cudam, CUstream stream);
	CUresult CopyCUDAToHost(CUDAMatrix& cudam, HOSTMatrix& hostm, CUstream stream);

}