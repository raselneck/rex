#include <rex/Math/Math.hxx>

REX_NS_BEGIN

#define MATH_PI          ( 3.14159265358979323846264338327950 )
#define MATH_TWO_PI      ( 6.28318530717958647692528676655900 )
#define MATH_PI_OVER_180 ( 0.01745329251994329576923690768489 )
#define MATH_INV_PI      ( 0.31830988618379067153776752674503 )
#define MATH_INV_TWO_PI  ( 0.15915494309189533576888376337251 )
#define MATH_EPSILON     ( 0.0001 )
#define MATH_HUGE_VALUE  ( 1.0E10 )



// get pi
real64 Math::Pi()
{
    return MATH_PI;
}

// get 2 * pi
real64 Math::TwoPi()
{
    return MATH_TWO_PI;
}

// get pi / 180
real64 Math::PiOver180()
{
    return MATH_PI_OVER_180;
}

// get 1 / pi
real64 Math::InvPi()
{
    return MATH_INV_PI;
}

// get 1 / ( 2 * pi )
real64 Math::InvTwoPi()
{
    return MATH_INV_TWO_PI;
}

// get a really small value
real64 Math::Epsilon()
{
    return MATH_EPSILON;
}

// get a huge value
real64 Math::HugeValue()
{
    return MATH_HUGE_VALUE;
}



// take 32-bit floor
int32 Math::Floor( real32 value )
{
    // derived from http://www.codeproject.com/Tips/700780/Fast-floor-ceiling-functions
    return static_cast<int32>( value + 1.0f ) - 1;
}

// take 32-bit ceiling
int32 Math::Ceiling( real32 value )
{
    // derived from http://www.codeproject.com/Tips/700780/Fast-floor-ceiling-functions
    return 1 - static_cast<int32>( 1.0f - value );
}

// take 64-bit floor
int64 Math::Floor( real64 value )
{
    return static_cast<int64>( value + 1.0 ) - 1;
}

// take 64-bit ceiling
int64 Math::Ceiling( real64 value )
{
    return 1 - static_cast<int64>( 1.0 - value );
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