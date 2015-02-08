#ifndef __REX_RANDOM_HXX
#define __REX_RANDOM_HXX
#pragma once

#include "Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Contains a way to generate pseudorandom numbers.
/// </summary>
class Random
{
    Random();
    Random( const Random& );
    ~Random();

public:
    /// <summary>
    /// Seeds the pseudorandom number generator.
    /// </summary>
    /// <param name=""></param>
    static void Seed( uint32 seed );

    /// <summary>
    /// Generates a random integer.
    /// </summary>
    static int32 RandInt();

    /// <summary>
    /// Generates a random real32.
    /// </summary>
    static real32 RandReal32();
};

REX_NS_END

#endif