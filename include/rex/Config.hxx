#pragma once

#include "CUDA.hxx"
#include <stdlib.h>


// TODO : Do we need to implement the device memory operators for pure virtual classes?


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
/// A macro for easily implementing a non-copyable class.
/// </summary>
/// <param name="cname">The class name.</param>
#define REX_NONCOPYABLE_CLASS(cname) \
    private: \
        cname( const cname& ) = delete; \
        cname( cname&& ) = delete; \
        cname& operator=( const cname& ) = delete; \
        cname& operator=( cname&& ) = delete;

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
        cname& operator=(cname&&) = delete;

/// <summary>
/// Implements the "new" and "delete" operators for the device.
/// </summary>
#define REX_IMPLEMENT_DEVICE_MEM_OPS() \
    public: \
        __device__ static void* operator new(size_t bytes) { return malloc( bytes ); } \
        __device__ static void  operator delete(void* mem) { free( mem ); } \
    private:



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

// Defining REX_MAX_PRECISION will use doubles instead of floats. Although they are far more
// accurate, they may drastically reduce the speed of Rex on the GPU. See here:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#from-graphics-processing-to-general-purpose-parallel-computing
// For alignment purposes, I also only recommend using REX_MAX_PRECISION with 64-bit builds.
#if defined( REX_MAX_PRECISION )
typedef real64 real_t;
#else
typedef real32 real_t;
#endif


// TODO : Look up build type detection with GCC
#if defined( _WIN64 )
typedef uint64 uint_t;
#else
typedef uint32 uint_t;
#endif



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