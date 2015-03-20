#pragma once

#include "../Config.hxx"
#include <iostream>
#include <mutex>

REX_NS_BEGIN

/// <summary>
/// Defines a thread-safe console logger.
/// </summary>
class Logger
{
    REX_STATIC_CLASS( Logger );

    static std::mutex _mutex;

public:
    /// <summary>
    /// Logs the given arguments to the console.
    /// </summary>
    /// <param name="args">The arguments to log.</param>
    template<typename ... Args> __host__ static void Log( const Args& ... args );
};

REX_NS_END

#include "Logger.inl"