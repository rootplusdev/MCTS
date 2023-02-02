#pragma once
#include<future>
#include<atomic>
class SpinLock
{
private:
	std::atomic_flag mylock = ATOMIC_FLAG_INIT;
public:
	SpinLock();
	void lock();
	void unlock();
};

