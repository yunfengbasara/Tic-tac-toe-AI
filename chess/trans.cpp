#include "trans.h"
#include <comutil.h>
using namespace util;

std::wstring util::UTF8ToUnicode(const std::string & str)
{
	int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
	std::wstring wstr(size_needed, 0);
	MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
	return wstr;
}

std::string util::UnicodeToANSI(const std::wstring & str)
{
	int size_needed = WideCharToMultiByte(CP_ACP, 0, &str[0], (int)str.size(), NULL, 0, NULL, NULL);
	std::string ansistr(size_needed, 0);
	WideCharToMultiByte(CP_ACP, 0, &str[0], (int)str.size(), &ansistr[0], size_needed, NULL, NULL);
	return ansistr;
}

std::string util::UTF8ToANSI(const std::string & str)
{
	return UnicodeToANSI(UTF8ToUnicode(str));
}

std::wstring util::ANSIToUnicode(const std::string & str)
{
	int size_needed = MultiByteToWideChar(CP_ACP, 0, &str[0], (int)str.size(), NULL, 0);
	std::wstring wstr(size_needed, 0);
	MultiByteToWideChar(CP_ACP, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
	return wstr;
}

std::string util::UnicodeToUTF8(const std::wstring & str)
{
	int size_needed = WideCharToMultiByte(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0, NULL, NULL);
	std::string utf8str(size_needed, 0);
	WideCharToMultiByte(CP_UTF8, 0, &str[0], (int)str.size(), &utf8str[0], size_needed, NULL, NULL);
	return utf8str;
}

std::string util::ANSIToUTF8(const std::string & str)
{
	return UnicodeToUTF8(ANSIToUnicode(str));
}