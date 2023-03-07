#include "type.h"
#include "util.h"
#include "cuda_runtime.h"

using namespace Eigen;
using namespace util;
using namespace std;

CUresult util::CreateCUDAMatrix(
    const HOSTMatrix& hostm, 
    CUDAMatrix& cudam)
{
    CUresult ret;

    cudam.width = hostm.cols();
    cudam.height = hostm.rows();
    cudam.stride = cudam.width * EZ;
    cudam.size = cudam.stride * cudam.height;

    ret = cuMemAllocPitch(
        &cudam.data, &cudam.pitch, 
        cudam.stride, cudam.height, EZ);

    cudam.pitchcols = cudam.pitch / EZ;

    return ret;
}

CUresult util::DestroyCUDAMatrix(
    CUDAMatrix& cudam)
{
    CUresult ret;

    ret = cuMemFree(cudam.data);
    cudam = CUDAMatrix();

    return ret;
}

CUresult util::CopyHostToCUDA(
    HOSTMatrix& hostm,
    CUDAMatrix& cudam, 
    CUstream stream)
{
    CUresult ret;

    CUDA_MEMCPY2D work;
    work.srcXInBytes = 0;
    work.srcY = 0;
    work.srcMemoryType = CU_MEMORYTYPE_HOST;
    work.srcHost = hostm.data();
    work.srcDevice = 0;
    work.srcArray = 0;
    work.srcPitch = hostm.cols() * EZ;

    work.dstXInBytes = 0;
    work.dstY = 0;
    work.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    work.dstHost = 0;
    work.dstDevice = cudam.data;
    work.dstArray = 0;
    work.dstPitch = cudam.pitch;

    work.WidthInBytes = cudam.stride;
    work.Height = cudam.height;

    ret = cuMemcpy2DAsync(&work, stream);

    return ret;
}

CUresult util::CopyCUDAToHost(
    CUDAMatrix& cudam, 
    HOSTMatrix& hostm, 
    CUstream stream)
{
    CUresult ret;

    CUDA_MEMCPY2D work;
    work.srcXInBytes = 0;
    work.srcY = 0;
    work.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    work.srcHost = 0;
    work.srcDevice = cudam.data;
    work.srcArray = 0;
    work.srcPitch = cudam.pitch;

    work.dstXInBytes = 0;
    work.dstY = 0;
    work.dstMemoryType = CU_MEMORYTYPE_HOST;
    work.dstHost = hostm.data();
    work.dstDevice = 0;
    work.dstArray = 0;
    work.dstPitch = hostm.cols() * EZ;

    work.WidthInBytes = cudam.stride;
    work.Height = cudam.height;

    ret = cuMemcpy2DAsync(&work, stream);

    return ret;
}