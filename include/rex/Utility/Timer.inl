#ifndef __REX_TIMER_INL
#define __REX_TIMER_INL

#include "Timer.hxx"

REX_NS_BEGIN

// check if running
inline bool Timer::IsRunning() const
{
    return _isRunning;
}

// get elapsed time in seconds
inline real64 Timer::GetElapsed() const
{
    Duration elapsed = _end - _start;
    return elapsed.count();
}

// start timer
inline void Timer::Start()
{
    _start = Clock::now();
}

// stop timer
inline void Timer::Stop()
{
    _end = Clock::now();
}

REX_NS_END

#endif