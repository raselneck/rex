#pragma once

#include "../Config.hxx"
#include <chrono>

REX_NS_BEGIN

/// <summary>
/// Defines a timer.
/// </summary>
class Timer
{
    typedef std::chrono::high_resolution_clock  Clock;
    typedef std::chrono::time_point<Clock>      TimePoint;
    typedef std::chrono::duration<real64>       Duration;

    TimePoint _start;
    TimePoint _end;
    bool _isRunning;

public:
    /// <summary>
    /// Creates a new timer.
    /// </summary>
    __host__ Timer();

    /// <summary>
    /// Destroys this timer.
    /// </summary>
    __host__ ~Timer();

    /// <summary>
    /// Checks to see if this timer is running.
    /// </summary>
    __host__ bool IsRunning() const;

    /// <summary>
    /// Gets the elapsed time, in seconds.
    /// </summary>
    __host__ real64 GetElapsed() const;

    /// <summary>
    /// Starts the timer.
    /// </summary>
    __host__ void Start();

    /// <summary>
    /// Stops the timer.
    /// </summary>
    __host__ void Stop();
};

REX_NS_END