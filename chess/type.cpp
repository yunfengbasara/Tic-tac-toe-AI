#include "type.h"
#include "util.h"
#include "cuda_runtime.h"

using namespace Eigen;
using namespace util;
using namespace std;

CUDAMatrix util::CreateCUDAMatrix(const HOSTMatrix& hostm)
{
    CUDAMatrix cudam;

    cudam = CreateCUDAMatrix(
        hostm.rows(), 
        hostm.cols());

    CopyHostToCUDA(hostm, cudam);

    return cudam;
}

CUDAMatrix util::CreateCUDAMatrix(size_t rows, size_t cols)
{
    CUDAMatrix cudam;

    cudam.rows = rows;
    cudam.cols = cols;
    cudam.size = rows * cols * EZ;

    checkCudaErrors(cuMemAlloc(&cudam.data, cudam.size));

    return cudam;
}

HOSTMatrix util::CreateHOSTMatrix(const CUDAMatrix& cudam)
{
    HOSTMatrix hostm = HOSTMatrix(cudam.rows, cudam.cols);

    CopyCUDAToHost(cudam, hostm);

    return hostm;
}

void util::DestroyCUDAMatrix(
    CUDAMatrix& cudam)
{
    cudam.height = 0;
    cudam.width = 0;
    cudam.size = 0;

    checkCudaErrors(cuMemFree(cudam.data));
    cudam.data = 0;
}

void util::CopyHostToCUDA(
    const HOSTMatrix& hostm,
    CUDAMatrix& cudam)
{
    checkCudaErrors(
        cuMemcpyHtoD(cudam.data, hostm.data(), cudam.size));
}

void util::CopyCUDAToHost(
    const CUDAMatrix& cudam, 
    HOSTMatrix& hostm)
{
    checkCudaErrors(
        cuMemcpyDtoH(hostm.data(), cudam.data, cudam.size));
}