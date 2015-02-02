#ifndef __REX_CONFIG_HXX
#define __REX_CONFIG_HXX
#pragma once

#define REX_NS_BEGIN namespace rex {
#define REX_NS_END   }

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

#include <string>
REX_NS_BEGIN

typedef std::string String;

REX_NS_END

#endif