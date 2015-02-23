#include <rex/Utility/Color.hxx>

REX_NS_BEGIN

static const real64 OneThird64 = 1.0 / 3.0;

const Color Color::Black    ( 0.0f );
const Color Color::White    ( 1.0f );
const Color Color::Red      ( 1.0f, 0.0f, 0.0f );
const Color Color::Green    ( 0.0f, 1.0f, 0.0f );
const Color Color::Blue     ( 0.0f, 0.0f, 1.0f );
const Color Color::Yellow   ( 1.0f, 1.0f, 0.0f );
const Color Color::Cyan     ( 0.0f, 1.0f, 1.0f );
const Color Color::Magenta  ( 1.0f, 0.0f, 1.0f );

// new color
Color::Color()
    : Color( 0.0f, 0.0f, 0.0f )
{
}

// new color
Color::Color( real32 all )
    : Color( all, all, all )
{
}

// new color
Color::Color( real32 r, real32 g, real32 b )
    : R( r ),
      G( g ),
      B( b )
{
}

// destroy color
Color::~Color()
{
    R = 0.0f;
    G = 0.0f;
    B = 0.0f;
}

REX_NS_END