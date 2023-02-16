#include "util.h"
#include <Windows.h>

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