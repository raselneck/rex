#ifndef __REX_LOGGER_HXX
#define __REX_LOGGER_HXX
#pragma once

#include "Config.hxx"
#include <iostream>
#include <fstream>
#include <sstream>

REX_NS_BEGIN

#pragma warning( push )
#pragma warning( disable : 4789 )

#define REX_LOGGER_WRITE \
    using List = int[]; \
    (void)List{ 0, ( (void)( _fout << args ), 0 ) ... }; \
    _fout.flush(); \
    (void)List{ 0, ( (void)( _cout << args ), 0 ) ... }

/// <summary>
/// Defines the static logger to use.
/// </summary>
class Logger
{
    static std::ofstream _fout;
    static std::ostream& _cout;

    Logger();
    Logger( const Logger& );
    Logger& operator=( const Logger& );

public:
    /// <summary>
    /// Logs the given arguments.
    /// </summary>
    /// <param name="args">The arguments to log.</param>
    template<typename ... Args> static void Log( const Args& ... args )
    {
        _fout << "[INFO] ";
        _cout << "[INFO] ";

        REX_LOGGER_WRITE;
    }

    /// <summary>
    /// Logs the given arguments as a warning.
    /// </summary>
    /// <param name="args">The arguments to log.</param>
    template<typename ... Args> static void LogWarning( const Args& ... args )
    {
        _fout << "[WARN] ";
        _cout << "[WARN] ";

        REX_LOGGER_WRITE;
    }

    /// <summary>
    /// Logs the given arguments as an error.
    /// </summary>
    /// <param name="args">The arguments to log.</param>
    template<typename ... Args> static void LogError( const Args& ... args )
    {
        _fout << "[ERROR] ";
        _cout << "[ERROR] ";

        REX_LOGGER_WRITE;
    }
};

#undef REX_LOGGER_WRITE
#pragma warning( pop )

REX_NS_END

#endif