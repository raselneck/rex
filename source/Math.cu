#include <rex/Math/Math.hxx>

REX_NS_BEGIN

// take 32-bit floor
int32 Math::Floor( real32 value )
{
    // derived from http://www.codeproject.com/Tips/700780/Fast-floor-ceiling-functions
    int32  c     = static_cast<int32>( Math::Abs( value ) ) + 1;
    int32  floor = static_cast<int32>( value + c ) - c;
    return floor;
}

// take 32-bit ceiling
int32 Math::Ceiling( real32 value )
{
    // derived from http://www.codeproject.com/Tips/700780/Fast-floor-ceiling-functions
    int32  c       = static_cast<int32>( Math::Abs( value ) ) + 1;
    int32  ceiling = c - static_cast<int32>( c - value );
    return ceiling;
}

// take 64-bit floor
int64 Math::Floor( real64 value )
{
    int64  c     = static_cast<int32>( Math::Abs( value ) ) + 1;
    int64  floor = static_cast<int64>( value + c ) - c;
    return floor;
}

// take 64-bit ceiling
int64 Math::Ceiling( real64 value )
{
    int32  c       = static_cast<int32>( Math::Abs( value ) ) + 1;
    int64  ceiling = c - static_cast<int64>( c - value );
    return ceiling;
}

// round real32
int32 Math::Round( real32 value )
{
    return ( value > 0.0f )
        ? Math::Floor( value + 0.5f )
        : Math::Ceiling( value - 0.5f );
}

// round real64
int64 Math::Round( real64 value )
{
    return ( value > 0.0 )
        ? Math::Floor( value + 0.5 )
        : Math::Ceiling( value - 0.5 );
}

REX_NS_END