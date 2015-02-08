#ifndef __REX_CONFIG_HXX
#define __REX_CONFIG_HXX
#pragma once

/**
 * NOTES:
 * 1. This ray tracer uses a right-handed coordinate system in its calculations.
 */

#define REX_NS_BEGIN namespace rex {
#define REX_NS_END   }

#define REX_DEFAULT_GAMMA (2.2f)

#if defined( NDEBUG )
#  define __RELEASE__ 1
#  define __DEBUG__   0
#else
#  define __RELEASE__ 0
#  define __DEBUG__   1
#endif

#include <stdint.h>
typedef int8_t   int8;
typedef int16_t  int16;
typedef int32_t  int32;
typedef int64_t  int64;
typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float    real32;
typedef double   real64;

#include <memory>
#include <string>

REX_NS_BEGIN

typedef std::string String;
template<class T> using Handle = std::shared_ptr<T>;

REX_NS_END

#endif