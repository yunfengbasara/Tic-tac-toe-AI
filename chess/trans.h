#pragma once
#include <string>
namespace util
{
	std::wstring UTF8ToUnicode(const std::string &str);
	std::string UnicodeToANSI(const std::wstring &str);
	std::string UTF8ToANSI(const std::string &str);

	std::wstring ANSIToUnicode(const std::string &str);
	std::string UnicodeToUTF8(const std::wstring &str);
	std::string ANSIToUTF8(const std::string &str);
};

