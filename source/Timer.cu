#include <rex/Utility/Timer.hxx>

REX_NS_BEGIN

// create timer
Timer::Timer()
: _isRunning( false )
{
}

// destroy timer
Timer::~Timer()
{
    _isRunning = 0;
}

// check if running
bool Timer::IsRunning() const
{
    return _isRunning;
}

// get elapsed time in seconds
real64 Timer::GetElapsed() const
{
    Duration elapsed = _end - _start;
    return elapsed.count();
}

// start timer
void Timer::Start()
{
    _start = Clock::now();
}

// stop timer
void Timer::Stop()
{
    _end = Clock::now();
}

REX_NS_END