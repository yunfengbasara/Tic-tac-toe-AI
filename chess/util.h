#pragma once
#include <functional>
#include <string>

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
}