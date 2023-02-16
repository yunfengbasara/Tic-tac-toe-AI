#pragma once
#include <functional>
#include <string>

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
}