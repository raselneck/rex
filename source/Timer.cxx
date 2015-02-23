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

REX_NS_END