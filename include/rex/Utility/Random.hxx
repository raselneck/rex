#ifndef __REX_RANDOM_HXX
#define __REX_RANDOM_HXX

#include "../Config.hxx"

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
    static int32 RandInt32();

    /// <summary>
    /// Generates a random integer.
    /// </summary>
    /// <param name="min">The (inclusive) minimum value.</param>
    /// <param name="max">The (exclusive) maximum value.</param>
    static int32 RandInt32( int32 min, int32 max );

    /// <summary>
    /// Generates a random real32.
    /// </summary>
    static real32 RandReal32();

    /// <summary>
    /// Generates a random real32.
    /// </summary>
    /// <param name="min">The (inclusive) minimum value.</param>
    /// <param name="max">The (exclusive) maximum value.</param>
    static real32 RandReal32( real32 min, real32 max );
};

REX_NS_END

#include "Random.inl"
#endif