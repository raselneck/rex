#pragma once

#include "CUDA.hxx"

/// <summary>
/// A macro for beginning the Rex namespace.
/// </summary>
#define REX_NS_BEGIN namespace rex {

/// <summary>
/// A macro for ending the Rex namespace.
/// </summary>
#define REX_NS_END   }

/// <summary>
/// A macro for getting the passed in item as a string.
/// </summary>
/// <param name="x">The item to "stringify."</param>
#define REX_STRINGIFY(x) #x

/// <summary>
/// A macro for getting the passed in item as a string, allowing for the passed in item to be a macro.
/// </summary>
/// <param name="x">The item to "stringify."</param>
#define REX_XSTRINGIFY(x) REX_STRINGIFY(x)

/// <summary>
/// A macro for easily implementing a non-movable class.
/// </summary>
/// <param name="cname">The class name.</param>
#define REX_NONMOVABLE_CLASS(cname) \
    private: \
        cname( cname&& ) = delete; \
        cname& operator=( cname&& ) = delete  /* no semicolon here to make the macro look like a function */

/// <summary>
/// A macro for easily implementing a non-copyable class.
/// </summary>
/// <param name="cname">The class name.</param>
#define REX_NONCOPYABLE_CLASS(cname) \
    private: \
        cname( const cname& ) = delete; \
        cname( cname&& ) = delete; \
        cname& operator=( const cname& ) = delete; \
        cname& operator=( cname&& ) = delete  /* no semicolon here to make the macro look like a function */

/// <summary>
/// A macro for easily implementing a static class.
/// </summary>
/// <param name="cname">The class name.</param>
#define REX_STATIC_CLASS(cname) \
    private: \
        cname() = delete; \
        cname(const cname&) = delete; \
        cname(cname&&) = delete; \
        ~cname() = delete; \
        cname& operator=(const cname&) = delete; \
        cname& operator=(cname&&) = delete /* no semicolon here to make the macro look like a function */



/// <summary>
/// A macro for easily getting the offset of a pointer.
/// </summary>
/// <param name="base">The base pointer.</param>
/// <param name="offs">The offset.</param>
#define REX_OFFSET(base, offs) static_cast<void*>( reinterpret_cast<uint8*>( base ) + ( offs ) )


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

/// <summary>
/// The string type used by Rex.
/// </summary>
typedef std::string String;

/// <summary>
/// The handle type used by Rex.
/// </summary>
template<typename T> using Handle = std::shared_ptr<T>;

REX_NS_END