#ifndef __REX_MATH_HXX
#define __REX_MATH_HXX
#pragma once

#include "Config.hxx"
#include "Color.hxx"
#include <math.h>

REX_NS_BEGIN

/// <summary>
/// Defines static math methods.
/// </summary>
struct Math
{
    /// <summary>
    /// Returns the minimum of the two given values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    template<class T> static inline T Min( T a, T b )
    {
        return ( a < b ) ? a : b;
    }

    /// <summary>
    /// Returns the maximum of the two given values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    template<class T> static inline T Max( T a, T b )
    {
        return ( a > b ) ? a : b;
    }

    /// <summary>
    /// Clamps the given value to the given min and max.
    /// </summary>
    /// <param name="value">The value to clamp</param>
    /// <param name="min">The minimum.</param>
    /// <param name="max">The maximum.</param>
    template<class T> static inline T Clamp( T value, T min, T max )
    {
        return Min( max, Max( min, value ) );
    }

    /// <summary>
    /// Linear interpolates between two values.
    /// </summary>
    /// <param name="v0">The first value.</param>
    /// <param name="v1">The second value.</param>
    /// <param name="t">The percentage to interpolate.</param>
    template<class T> static inline T Lerp( T v0, T v1, T t )
    {
        return ( T( 1 ) - t ) * v0 + t * v1;
    }

    /// <summary>
    /// Raises the given color to the given exponent.
    /// </summary>
    /// <param name="color">The color to use as a base.</param>
    /// <param name="exponent">The exponent to raise to.</param>
    static inline Color Pow( const Color& color, real32 exponent )
    {
        
    }
};

REX_NS_END

#endif