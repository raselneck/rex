#ifndef __REX_MATH_INL
#define __REX_MATH_INL

#include "Math.hxx"

// Floor and Ceiling functions derived from http://www.codeproject.com/Tips/700780/Fast-floor-ceiling-functions

REX_NS_BEGIN

// take floor
inline int32 Math::Floor( real32 value )
{
    return (int32)( value + 64.0f ) - 64;
}

// take ceiling
inline int32 Math::Ceiling( real32 value )
{
    return 64 - (int32)( 64.0f - value );
}

// take floor
inline int64 Math::Floor( real64 value )
{
    return (int64)( value + 64.0 ) - 64;
}

// take ceiling
inline int64 Math::Ceiling( real64 value )
{
    return 64 - (int64)( 64.0 - value );
}

// round real32
inline int32 Math::Round( real32 value )
{
    return ( value > 0.0f )
        ? Math::Floor  ( value + 0.5f )
        : Math::Ceiling( value - 0.5f );
}

// round real64
inline int64 Math::Round( real64 value )
{
    return ( value > 0.0 )
        ? Math::Floor  ( value + 0.5 )
        : Math::Ceiling( value - 0.5 );
}

// get minimum
template<class T> inline T Math::Min( T val1, T val2 )
{
    return ( val1 < val2 ) ? val1 : val2;
}

// get maximum
template<class T> inline T Math::Max( T val1, T val2 )
{
    return ( val1 > val2 ) ? val1 : val2;
}

// clamps to range
template<class T> inline T Math::Clamp( T value, T min, T max )
{
    return Min( max, Max( min, value ) );
}

// linearly interpolate value
template<class T> inline T Math::Lerp( T v0, T v1, T t )
{
    return ( T( 1 ) - t ) * v0 + t * v1;
}

// map from one range to another
template<class T> inline T Math::Map( T value, T min1, T max1, T min2, T max2 )
{
    // derived from http://stackoverflow.com/questions/5731863/mapping-a-numeric-range-onto-another

    real64 slope = 1.0 * ( max2 - min2 ) / ( max1 - min1 );
    real64 out = min2 + slope * ( value - min1 );
    return static_cast<T>( out );
}

REX_NS_END

#endif