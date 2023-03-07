#pragma once
#include <functional>
#include <string>
#include <vector>
#include "cuda_runtime.h"
#include "cuda.h"

namespace util
{
	// �ӳ�ִ��
	class Defer
	{
	public:
		Defer(std::function<void()>&& pfun);
		~Defer();
	private:
		std::function<void()> m_pFunc;
	};
}

// ������ʱ����
#define defer(code) util::Defer __([&]{code})

namespace util
{
	// ��ȡ��ǰ����·��
	std::wstring GetCurrentDir();

	// ����CU
	bool CompileFileToCUBIN(const std::wstring& cufile, std::vector<char>& cubin);

	// ѡ��CUDA�豸
	bool FindCudaDeviceDRV(CUdevice& device);

	// ѡ�����Gflops/s�豸���
	bool GpuGetMaxGflopsDeviceIdDRV(int& id);

	// GPU��ϵ�⹹
	int ConvertSMVer2CoresDRV(int major, int minor);

	// ��ȡ���CUDA�豸�汾
	bool FindCudaVersion(CUdevice& device, int& major, int& minor);

	// ��ȡcubin
	bool LoadCUBIN(const std::vector<char>& cubin, CUmodule& module);
}