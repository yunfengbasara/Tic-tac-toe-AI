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

CUDAMatrix util::CreateCUDAMatrix(int h, int w)
{
    CUDAMatrix cudam;

    cudam.height = h;
    cudam.width = w;
    cudam.stride = cudam.width * EZ;
    cudam.size = cudam.stride * cudam.height;

    cuMemAllocPitch(
        &cudam.data, &cudam.pitch,
        cudam.stride, cudam.height, EZ);

    cudam.pitchwidth = cudam.pitch / EZ;

    return cudam;
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
    const HOSTMatrix& hostm,
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

    if (stream == nullptr) {
        ret = cuMemcpy2D(&work);
    }
    else {
        ret = cuMemcpy2DAsync(&work, stream);
    }

    return ret;
}

CUresult util::CopyCUDAToHost(
    const CUDAMatrix& cudam, 
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

    if (stream == nullptr) {
        ret = cuMemcpy2D(&work);
    }
    else {
        ret = cuMemcpy2DAsync(&work, stream);
    }

    return ret;
}