#ifndef __REX_RANDOM_INL
#define __REX_RANDOM_INL

#include "Random.hxx"
#include "Math.hxx"
#include <stdlib.h>
#include <time.h>

#if __DEBUG__
#  include "Debug.hxx"
#endif

REX_NS_BEGIN

// seed PRNG
inline void Random::Seed( uint32 seed )
{
    srand( seed );
}

// generate random int
inline int32 Random::RandInt32()
{
    return rand();
}

// generate random int
inline int32 Random::RandInt32( int32 min, int32 max )
{
    return static_cast<int32>( RandReal32( 0.0f, static_cast<real32>( max - min + 1 ) ) + min );
}

// generate random float
inline real32 Random::RandReal32()
{
    real32 value = static_cast<real32>( rand() ) / static_cast<real32>( RAND_MAX );

#if __DEBUG__
    REX_ASSERT( value >= 0.0f && value <= 1.0f, "Random range check value failed" );
#endif

    return value;
}

// generate random float
inline real32 Random::RandReal32( real32 min, real32 max )
{
    return RandReal32() * ( max - min ) + min;
}

REX_NS_END

#endif