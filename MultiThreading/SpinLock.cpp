#include "SpinLock.h"

SpinLock::SpinLock() {}

void SpinLock::lock()
{
	while (mylock.test_and_set(std::memory_order_acquire));
}

void SpinLock::unlock()
{
	mylock.clear(std::memory_order_release);
}