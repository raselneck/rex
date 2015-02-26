#ifndef __REX_CONFIG_HXX
#define __REX_CONFIG_HXX

/**
 * "Trick or bear?"
 * "Bear??"
 * "HE HAS CHOSEN THE BEAR!"
 * *sounds of chains and snarling off in the distance*
 */

#define REX_NS_BEGIN namespace rex {
#define REX_NS_END   }

#define REX_DEFAULT_GAMMA   (  2.2f )
#define REX_DEFAULT_SAMPLES (  1    )
#define REX_DEFAULT_SETS    ( 83    )

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

#if !defined( REX_CUDA_BUILD )

/// <summary>
/// The handle type used by Rex for managing pointers.
/// </summary>
template<class T> using Handle = std::shared_ptr<T>;

#endif

REX_NS_END

#endif