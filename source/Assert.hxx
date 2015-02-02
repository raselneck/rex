#ifndef __REX_ASSERT_HXX
#define __REX_ASSERT_HXX
#pragma once

#include "Config.hxx"
#include "Logger.hxx"

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