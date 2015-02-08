#ifndef __REX_RANDOM_INL
#define __REX_RANDOM_INL
#pragma once

#include "Random.hxx"
#include <stdlib.h>
#include <time.h>

REX_NS_BEGIN

// seed PRNG
inline void Random::Seed( uint32 seed )
{
    srand( seed );
}

// generate random int
inline int32 Random::RandInt()
{
    return rand();
}

// generate random float
inline real32 Random::RandReal32()
{
    return static_cast<real32>( rand() ) / static_cast<real32>( RAND_MAX );
}

REX_NS_END

#endif