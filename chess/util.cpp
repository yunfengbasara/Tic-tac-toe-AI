#include "util.h"
#include <Windows.h>
#include <filesystem>
#include <iostream>
#include "nvrtc.h"
#include "trans.h"

using namespace util;

Defer::Defer(std::function<void()>&& pfun)
    : m_pFunc(pfun) {

}

Defer::~Defer() {
    m_pFunc();
}

std::wstring util::GetCurrentDir()
{
	TCHAR szExePath[MAX_PATH] = { 0 };
	GetModuleFileName(NULL, szExePath, MAX_PATH);
	std::wstring strExeDir = szExePath;
	size_t offset = strExeDir.rfind(L"\\");
	strExeDir = strExeDir.substr(0, offset + 1);
	return strExeDir;
}

bool util::CompileFileToCUBIN(const std::wstring& cufile, std::vector<char>& cubin)
{
	// 当前工作目录而非exe运行时目录
	std::wstring sourcepath = std::filesystem::current_path();
	sourcepath += L"\\";

	// cuda文件
	std::wstring cupath = sourcepath + cufile;

	HANDLE hFile = CreateFile(cupath.c_str(),
		GENERIC_READ | GENERIC_WRITE, 0, NULL,
		OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) {
		return false;
	}

	DWORD size = 0;
	size = ::GetFileSize(hFile, NULL);

	std::vector<char> context(size + 1, 0);
	if (!::ReadFile(hFile, (LPBYTE)&context[0], size, &size, NULL)) {
		CloseHandle(hFile);
		return false;
	}

	CloseHandle(hFile);

	// 编译过程
	nvrtcResult res;
	nvrtcProgram prog;
	res = nvrtcCreateProgram(&prog, context.data(), NULL, 0, NULL, NULL);
	if (res != NVRTC_SUCCESS) {
		return false;
	}

	// 编译显卡选项
	CUdevice cudev;
	int major = 0, minor = 0;
	if (!FindCudaVersion(cudev, major, minor)) {
		return false;
	}

	std::wstring archOptions = L"--gpu-architecture=sm_";
	archOptions += std::to_wstring(major);
	archOptions += std::to_wstring(minor);
	std::string op1 = UnicodeToANSI(archOptions);

	char* compileParams[1];
	compileParams[0] = (char*)malloc(op1.length() + 1);
	compileParams[0][op1.length()] = 0;
	memcpy(compileParams[0], op1.c_str(), op1.length());

	res = nvrtcCompileProgram(prog, 1, compileParams);
	free(compileParams[0]);

	// 编译错误日志
	if (res != NVRTC_SUCCESS) {
		size_t logSize;
		nvrtcGetProgramLogSize(prog, &logSize);
		std::vector<char> log(logSize + 1, 0);
		nvrtcGetProgramLog(prog, &log[0]);
		std::cout << "cuda compilation log ---" << std::endl;
		std::cout << log.data() << std::endl;
		std::cout << "end log ---" << std::endl;
		return false;
	}

	// 获取编译文件
	size_t codeSize;
	res = nvrtcGetCUBINSize(prog, &codeSize);
	if (res != NVRTC_SUCCESS) {
		return false;
	}

	std::vector<char> code(codeSize);
	res = nvrtcGetCUBIN(prog, &code[0]);
	if (res != NVRTC_SUCCESS) {
		return false;
	}

	cubin = move(code);
	return true;
}

bool util::FindCudaDeviceDRV(CUdevice& device)
{
	int devID = 0;
	if (!GpuGetMaxGflopsDeviceIdDRV(devID)) {
		return false;
	}

	if (cuDeviceGet(&device, devID) != CUDA_SUCCESS) {
		return false;
	}

	char name[100] = { 0 };
	if (cuDeviceGetName(name, 100, device) != CUDA_SUCCESS) {
		return false;
	}

	return true;
}

bool util::GpuGetMaxGflopsDeviceIdDRV(int& id)
{
	CUdevice current_device = 0;
	CUdevice max_perf_device = 0;
	int device_count = 0;
	int sm_per_multiproc = 0;
	unsigned long long max_compute_perf = 0;
	int major = 0;
	int minor = 0;
	int multiProcessorCount;
	int clockRate;
	int devices_prohibited = 0;

	if (cuInit(0) != CUDA_SUCCESS) {
		return false;
	}

	if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS) {
		return false;
	}

	if (device_count == 0) {
		return false;
	}

	while (current_device < device_count) {
		if (cuDeviceGetAttribute(&multiProcessorCount, 
			CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
			current_device) != CUDA_SUCCESS) {
			return false;
		}

		if (cuDeviceGetAttribute(&clockRate, 
			CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 
			current_device) != CUDA_SUCCESS) {
			return false;
		}

		if (cuDeviceGetAttribute(&major, 
			CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
			current_device) != CUDA_SUCCESS) {
			return false;
		}

		if (cuDeviceGetAttribute(&minor, 
			CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 
			current_device) != CUDA_SUCCESS) {
			return false;
		}

		int computeMode;
		if (cuDeviceGetAttribute(&computeMode,
			CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
			current_device) != CUDA_SUCCESS) {
			return false;
		}

		if (computeMode != CU_COMPUTEMODE_PROHIBITED) {
			if (major == 9999 && minor == 9999) {
				sm_per_multiproc = 1;
			}
			else {
				sm_per_multiproc = ConvertSMVer2CoresDRV(major, minor);
			}

			unsigned long long compute_perf = (unsigned long long)(
				multiProcessorCount * sm_per_multiproc * clockRate);

			if (compute_perf > max_compute_perf) {
				max_compute_perf = compute_perf;
				max_perf_device = current_device;
			}
		}
		else {
			devices_prohibited++;
		}

		++current_device;
	}

	if (devices_prohibited == device_count) {
		return false;
	}

	id = max_perf_device;
	return true;
}

int util::ConvertSMVer2CoresDRV(int major, int minor)
{
	typedef struct {
		int SM;
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{0x80,  64},
		{0x86, 128},
		{0x87, 128},
		{0x90, 128},
		{-1, -1} 
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	return nGpuArchCoresPerSM[index - 1].Cores;
}

bool util::FindCudaVersion(CUdevice& device, int& major, int& minor)
{
	if (!FindCudaDeviceDRV(device)) {
		return false;
	}

	if (cuDeviceGetAttribute(&major,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		device) != CUDA_SUCCESS) {
		return false;
	}

	if (cuDeviceGetAttribute(&minor,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		device) != CUDA_SUCCESS) {
		return false;
	}

	return true;
}

bool util::LoadCUBIN(const std::vector<char>& cubin, CUmodule& module)
{
	CUdevice cudev;
	int major = 0, minor = 0;
	if (!FindCudaVersion(cudev, major, minor)) {
		return false;
	}

	if (cuInit(0) != CUDA_SUCCESS) {
		return false;
	}

	CUcontext context;
	if (cuCtxCreate(&context, 0, cudev) != CUDA_SUCCESS) {
		return false;
	}

	if (cuModuleLoadData(&module, cubin.data()) != CUDA_SUCCESS) {
		return false;
	}

	return true;
}

void util::CheckCudaErrors(CUresult err, const char* file, const int line) {
	if (CUDA_SUCCESS == err) {
		return;
	}

	const char* errorStr = NULL;
	cuGetErrorString(err, &errorStr);
	fprintf(stderr,
		"CheckCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
		"line %i.\n",
		err, errorStr, file, line);
	exit(EXIT_FAILURE);
}

void util::CheckCuBlasErrors(cublasStatus_t err, const char* file, const int line) {
	if (CUBLAS_STATUS_SUCCESS == err) {
		return;
	}

	fprintf(stderr,
		"CheckCuBlasErrors() Driver API error = %04d from file <%s>, "
		"line %i.\n",
		err, file, line);
	exit(EXIT_FAILURE);
}