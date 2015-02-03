#ifndef __REX_ASSERT_HXX
#define __REX_ASSERT_HXX
#pragma once

#include "Config.hxx"
#include "Logger.hxx"

/// <summary>
/// Custom assert macro for Rex.
/// </summary>
/// <param name="cond">The condition to check.</param>
/// <param name="message">The message to log if the condition fails.</param>
#define RexAssert(cond, message) { \
        bool __result = ( cond ); \
        if ( !__result ) { \
            rex::String __fname = __FILE__; \
            __fname = __fname.substr( __fname.find_last_of( '\\' ) + 1 ); \
            rex::Logger::LogError( "(", __fname, ":", __LINE__, ") Assertion failed: ", message, "\n  ", #cond, "\n" ); \
            __debugbreak(); \
        } \
    }

#endif