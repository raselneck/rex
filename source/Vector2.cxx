#include "Vector2.hxx"
#include "Vector2.inl"
#include <math.h>

REX_NS_BEGIN

// new 2D vector
Vector2::Vector2()
    : Vector2( 0.0, 0.0 )
{
}

// new 2D vector
Vector2::Vector2( real64 all )
    : Vector2( all, all )
{
}

// new 2D vector
Vector2::Vector2( real64 x, real64 y )
    : X( x ),
      Y( y )
{
}

// destroy 2D vector
Vector2::~Vector2()
{
    X = Y = 0.0;
}

REX_NS_END