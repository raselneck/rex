#ifndef __REX_MATH_HXX
#define __REX_MATH_HXX

#include "Config.hxx"
#include <math.h>

#define REX_INT32_MAX  (  2147483647  )
#define REX_INT32_MIN  ( -2147483648  )
#define REX_UINT32_MAX (  4294967296U )
#define REX_UINT32_MIN (           0U )

#define REX_INT64_MAX  (  9223372036854775807  )
#define REX_INT64_MIN  ( -9223372036854775808  )
#define REX_UINT64_MAX ( 18446744073709551616U )
#define REX_UINT64_MIN (                    0U )

REX_NS_BEGIN

/// <summary>
/// Defines static math methods.
/// </summary>
class Math
{
    Math();
    Math( const Math& );
    ~Math();

public:
    static const real64 PI;
    static const real64 TWO_PI;
    static const real64 PI_OVER_180;
    static const real64 INV_PI;
    static const real64 INV_TWO_PI;

    static const real64 EPSILON;
    static const real64 HUGE_VALUE;
    static const real64 ONE_THIRD;

    /// <summary>
    /// Returns the floor of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    static int32 Floor( real32 value );

    /// <summary>
    /// Returns the ceiling of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    static int32 Ceiling( real32 value );

    /// <summary>
    /// Returns the floor of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    static int64 Floor( real64 value );

    /// <summary>
    /// Returns the ceiling of the given value.
    /// </summary>
    /// <param name="value">The value.</param>
    static int64 Ceiling( real64 value );

    /// <summary>
    /// Rounds the given value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    static int32 Round( real32 value );

    /// <summary>
    /// Rounds the given value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    static int64 Round( real64 value );

    /// <summary>
    /// Returns the minimum of the two given values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    template<class T> static T Min( T a, T b );

    /// <summary>
    /// Returns the maximum of the two given values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    template<class T> static T Max( T a, T b );

    /// <summary>
    /// Clamps the given value to the given min and max.
    /// </summary>
    /// <param name="value">The value to clamp</param>
    /// <param name="min">The minimum.</param>
    /// <param name="max">The maximum.</param>
    template<class T> static T Clamp( T value, T min, T max );

    /// <summary>
    /// Maps the given value in the first range to a value in the second range.
    /// </summary>
    /// <param name="value">The value to map.</param>
    /// <param name="min1">The minimum for the first range.</param>
    /// <param name="max1">The maximum for the first range.</param>
    /// <param name="min2">The minimum for the second range.</param>
    /// <param name="max2">The maximum for the second range.</param>
    template<class T> static T Map( T value, T min1, T max1, T min2, T max2 );

    /// <summary>
    /// Linear interpolates between two values.
    /// </summary>
    /// <param name="v0">The first value.</param>
    /// <param name="v1">The second value.</param>
    /// <param name="t">The percentage to interpolate.</param>
    template<class T> static T Lerp( T v0, T v1, T t );
};

REX_NS_END

#include "Math.inl"
#endif