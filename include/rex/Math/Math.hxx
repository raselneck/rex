#pragma once

#include "../Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines static math methods.
/// </summary>
class Math
{
    REX_STATIC_CLASS( Math );

public:
    /// <summary>
    /// Gets the constant Pi.
    /// </summary>
    __cuda_func__ static real64 Pi();

    /// <summary>
    /// Gets the constant 2 * Pi.
    /// </summary>
    __cuda_func__ static real64 TwoPi();

    /// <summary>
    /// Gets the constant Pi / 180.
    /// </summary>
    __cuda_func__ static real64 PiOver180();

    /// <summary>
    /// Gets the constant 1 / Pi.
    /// </summary>
    __cuda_func__ static real64 InvPi();

    /// <summary>
    /// Gets the constant 1 / ( 2 * Pi ).
    /// </summary>
    __cuda_func__ static real64 InvTwoPi();

    /// <summary>
    /// Gets a really small value.
    /// </summary>
    __cuda_func__ static real64 Epsilon();

    /// <summary>
    /// Gets a really big value.
    /// </summary>
    __cuda_func__ static real64 HugeValue();

    /// <summary>
    /// Returns the floor of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    __cuda_func__ static int32 Floor( real32 value );

    /// <summary>
    /// Returns the ceiling of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    __cuda_func__ static int32 Ceiling( real32 value );

    /// <summary>
    /// Returns the floor of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    __cuda_func__ static int64 Floor( real64 value );

    /// <summary>
    /// Returns the ceiling of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    __cuda_func__ static int64 Ceiling( real64 value );

    /// <summary>
    /// Rounds the given value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    __cuda_func__ static int32 Round( real32 value );

    /// <summary>
    /// Rounds the given value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    __cuda_func__ static int64 Round( real64 value );

    /// <summary>
    /// Returns the minimum of the two given values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    template<class T> __cuda_func__ static T Min( T a, T b );

    /// <summary>
    /// Returns the maximum of the two given values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    template<class T> __cuda_func__ static T Max( T a, T b );

    /// <summary>
    /// Clamps the given value to the given min and max.
    /// </summary>
    /// <param name="value">The value to clamp</param>
    /// <param name="min">The minimum.</param>
    /// <param name="max">The maximum.</param>
    template<class T> __cuda_func__ static T Clamp( T value, T min, T max );

    /// <summary>
    /// Maps the given value in the first range to a value in the second range.
    /// </summary>
    /// <param name="value">The value to map.</param>
    /// <param name="min1">The minimum for the first range.</param>
    /// <param name="max1">The maximum for the first range.</param>
    /// <param name="min2">The minimum for the second range.</param>
    /// <param name="max2">The maximum for the second range.</param>
    template<class T> __cuda_func__ static T Map( T value, T min1, T max1, T min2, T max2 );

    /// <summary>
    /// Linear interpolates between two values.
    /// </summary>
    /// <param name="v0">The first value.</param>
    /// <param name="v1">The second value.</param>
    /// <param name="t">The percentage to interpolate.</param>
    template<class T> __cuda_func__ static T Lerp( T v0, T v1, T t );
};

REX_NS_END

#include "Math.inl"