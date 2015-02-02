#include "Color.hxx"

REX_NS_BEGIN

Color::Color()
    : Color( 0.0f, 0.0f, 0.0f )
{
}

Color::Color( real32 all )
    : Color( all, all, all )
{
}

Color::Color( real32 r, real32 g, real32 b )
    : R( r ),
      G( g ),
      B( b )
{
}

Color::~Color()
{
    R = 0.0f;
    G = 0.0f;
    B = 0.0f;
}

REX_NS_END