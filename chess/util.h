#pragma once
#include <functional>
#include <string>
#include <vector>
#include "cuda_runtime.h"
#include "cuda.h"

namespace util
{
	// 延迟执行
	class Defer
	{
	public:
		Defer(std::function<void()>&& pfun);
		~Defer();
	private:
		std::function<void()> m_pFunc;
	};
}

// 定义临时对象
#define defer(code) util::Defer __([&]{code})

namespace util
{
	// 获取当前运行路径
	std::wstring GetCurrentDir();

	// 编译CU
	bool CompileFileToCUBIN(const std::wstring& cufile, std::vector<char>& cubin);

	// 选择CUDA设备
	bool FindCudaDeviceDRV(CUdevice& device);

	// 选择最高Gflops/s设备编号
	bool GpuGetMaxGflopsDeviceIdDRV(int& id);

	// GPU体系解构
	int ConvertSMVer2CoresDRV(int major, int minor);

	// 获取最佳CUDA设备版本
	bool FindCudaVersion(CUdevice& device, int& major, int& minor);

	// 读取cubin
	bool LoadCUBIN(const std::vector<char>& cubin, CUmodule& module);
}