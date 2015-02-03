#include "Color.hxx"
#include "Vector3.hxx"

REX_NS_BEGIN

static const real64 OneThird64 = 1.0 / 3.0;

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

// get average
real32 Color::GetAverage() const
{
    return real32( ( R + G + B ) * OneThird64 );
}

Color::operator Vector3()
{
    return Vector3( R, G, B );
}

bool Color::operator==( const Color& c ) const
{
    return R == c.R
        && G == c.G
        && B == c.B;
}

bool Color::operator!=( const Color& c ) const
{
    return R != c.R
        && G != c.G
        && B != c.B;
}

Color Color::operator+( const Color& c ) const
{
    Color result(
        R + c.R,
        G + c.G,
        B + c.B
    );
    return result;
}

Color Color::operator-( const Color& c ) const
{
    Color result(
        R - c.R,
        G - c.G,
        B - c.B
    );
    return result;
}

Color Color::operator*( const Color& c ) const
{
    Color result(
        R * c.R,
        G * c.G,
        B * c.B
    );
    return result;
}

Color Color::operator/( const Color& c ) const
{
    Color result(
        R / c.R,
        G / c.G,
        B / c.B
    );
    return result;
}

Color& Color::operator+=( const Color& c )
{
    R += c.R;
    G += c.G;
    B += c.B;
    return *this;
}

Color& Color::operator-=( const Color& c )
{
    R -= c.R;
    G -= c.G;
    B -= c.B;
    return *this;
}

Color& Color::operator*=( const Color& c )
{
    R *= c.R;
    G *= c.G;
    B *= c.B;
    return *this;
}

Color& Color::operator/=( const Color& c )
{
    R /= c.R;
    G /= c.G;
    B /= c.B;
    return *this;
}

Color operator*( const Color& c, real32 s )
{
    Color result(
        c.R * s,
        c.G * s,
        c.B * s
    );
    return result;
}

Color operator*( real32 s, const Color& c )
{
    Color result(
        c.R * s,
        c.G * s,
        c.B * s
    );
    return result;
}

Color& operator*=( Color& c, real32 s )
{
    c.R *= s;
    c.G *= s;
    c.B *= s;
    return c;
}

Color& operator*=( real32 s, Color& c )
{
    c.R *= s;
    c.G *= s;
    c.B *= s;
    return c;
}

REX_NS_END