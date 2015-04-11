#pragma once

#include "../Config.hxx"
#include <iostream>
#include <mutex>

#define REX_DEBUG_LOG(...) rex::Logger::Log( "(", rex::Logger::GetAbsoluteFileName( __FILE__ ), ":", __LINE__, ") ", __VA_ARGS__ )

REX_NS_BEGIN

/// <summary>
/// Defines a thread-safe console logger.
/// </summary>
class Logger
{
    REX_STATIC_CLASS( Logger );

    static std::mutex _mutex;

    /// <summary>
    /// Converts the given value into a string.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    template<typename T> static __host__ String ToString( const T& value );

    /// <summary>
    /// Merges a given list of strings.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    __host__ static String Merge( std::initializer_list<String> list );

public:
    /// <summary>
    /// Gets the absolute file name for the given file path.
    /// <summary>
    /// <param name="fname">The file name.</param>
    static String GetAbsoluteFileName( const char* fname );

    /// <summary>
    /// Logs the given arguments to the console.
    /// </summary>
    /// <param name="args">The arguments to log.</param>
    template<typename ... Args> __host__ static void Log( const Args& ... args );
};

REX_NS_END

#include "Logger.inl"