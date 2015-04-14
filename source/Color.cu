#include <rex/Graphics/Color.hxx>
#include <rex/Math/Math.hxx>
#include <math.h>

REX_NS_BEGIN

// create a color
Color::Color()
    : Color( 0.0f, 0.0f, 0.0f )
{
}

// create a color w/ value for all
Color::Color( real_t all )
    : Color( all, all, all )
{
}

// create a color w/ r, g, and b
Color::Color( real_t r, real_t g, real_t b )
    : R( r ),
      G( g ),
      B( b )
{
}

// destroy a color
Color::~Color()
{
    R = 0.0f;
    G = 0.0f;
    B = 0.0f;
}

// linearly interpolate two colors
Color Color::Lerp( const Color& c1, const Color& c2, real_t amount )
{
    return Color( Math::Lerp( c1.R, c2.R, amount ),
                  Math::Lerp( c1.G, c2.G, amount ),
                  Math::Lerp( c1.B, c2.B, amount ) );
}

// darken a color
Color Color::Darken( const Color& color, real_t amount )
{
    return Color::Lerp( color, Color::Black(), amount );
}

// lighten a color
Color Color::Lighten( const Color& color, real_t amount )
{
    return Color::Lerp( color, Color::White(), amount );
}

// raise a color to a power
Color Color::Pow( const Color& color, real_t exp )
{
    return Color( pow( color.R, exp ),
                  pow( color.G, exp ),
                  pow( color.B, exp ) );
}

#pragma region Pre-defined Colors

// get red
Color Color::Red()
{
    return Color( 1.0f, 0.0f, 0.0f );
}

// get blue
Color Color::Blue()
{
    return Color( 0.0f, 0.0f, 1.0f );
}

// get green
Color Color::Green()
{
    return Color( 0.0f, 1.0f, 0.0f );
}

// get magenta
Color Color::Magenta()
{
    return Color( 1.0f, 0.0f, 1.0f );
}

// get yellow
Color Color::Yellow()
{
    return Color( 1.0f, 1.0f, 0.0f );
}

// get cyan
Color Color::Cyan()
{
    return Color( 0.0f, 1.0f, 1.0f );
}

// get orange
Color Color::Orange()
{
    return Color( 1.0f, 0.5f, 0.0f );
}

// get purple
Color Color::Purple()
{
    return Color( 0.5f, 0.0f, 1.0f );
}

// get white
Color Color::White()
{
    return Color( 1.0f );
}

// get black
Color Color::Black()
{
    return Color( 0.0f );
}

#pragma endregion

#pragma region Operators

bool Color::operator==( const Color& c ) const
{
    return ( R == c.R )
        && ( G == c.G )
        && ( B == c.B );
}

bool Color::operator!=( const Color& c ) const
{
    return !( *this == c );
}

Color Color::operator+( const Color& c ) const
{
    return Color( R + c.R,
                  G + c.G,
                  B + c.B );
}

Color Color::operator-( const Color& c ) const
{
    return Color( R - c.R,
                  G - c.G,
                  B - c.B );
}

Color Color::operator/( real_t s ) const
{
    return Color( R / s,
                  G / s,
                  B / s );
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

Color& Color::operator*=( real_t s )
{
    R *= s;
    G *= s;
    B *= s;
    return *this;
}

Color& Color::operator/=( real_t s )
{
    R /= s;
    G /= s;
    B /= s;
    return *this;
}

Color operator*( const Color& c1, const Color& c2 )
{
    return Color( c1.R * c2.R,
                  c1.G * c2.G,
                  c1.B * c2.B );
}

Color operator*( const Color& c, real_t s )
{
    return Color( c.R * s,
                  c.G * s,
                  c.B * s );
}

Color operator*( real_t s, const Color& c )
{
    return Color( c.R * s,
                  c.G * s,
                  c.B * s );
}

std::ostream& operator<<( std::ostream& stream, const Color& color )
{
    stream << "{color : " << color.R << ", " << color.G << ", " << color.B << "}";

    return stream;
}

#pragma endregion

REX_NS_END