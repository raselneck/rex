#ifndef __REX_COLOR_INL
#define __REX_COLOR_INL

#include "Color.hxx"

REX_NS_BEGIN

// get average
inline real32 Color::GetAverage() const
{
    return real32( ( R + G + B ) * Math::ONE_THIRD );
}

// raise color to a power
inline Color Color::Pow( const Color& color, real32 exp )
{
    Color out;
    out.R = pow( color.R, exp );
    out.G = pow( color.G, exp );
    out.B = pow( color.B, exp );
    return out;
}

inline bool Color::operator==( const Color& c ) const
{
    return R == c.R
        && G == c.G
        && B == c.B;
}

inline bool Color::operator!=( const Color& c ) const
{
    return R != c.R
        && G != c.G
        && B != c.B;
}

inline Color Color::operator+( const Color& c ) const
{
    Color result(
        R + c.R,
        G + c.G,
        B + c.B
    );
    return result;
}

inline Color Color::operator-( const Color& c ) const
{
    Color result(
        R - c.R,
        G - c.G,
        B - c.B
    );
    return result;
}

inline Color Color::operator*( const Color& c ) const
{
    Color result(
        R * c.R,
        G * c.G,
        B * c.B
    );
    return result;
}

inline Color Color::operator/( const Color& c ) const
{
    Color result(
        R / c.R,
        G / c.G,
        B / c.B
    );
    return result;
}

inline Color& Color::operator+=( const Color& c )
{
    R += c.R;
    G += c.G;
    B += c.B;
    return *this;
}

inline Color& Color::operator-=( const Color& c )
{
    R -= c.R;
    G -= c.G;
    B -= c.B;
    return *this;
}

inline Color& Color::operator*=( const Color& c )
{
    R *= c.R;
    G *= c.G;
    B *= c.B;
    return *this;
}

inline Color& Color::operator/=( const Color& c )
{
    R /= c.R;
    G /= c.G;
    B /= c.B;
    return *this;
}

inline Color operator*( const Color& c, real32 s )
{
    Color result(
        c.R * s,
        c.G * s,
        c.B * s
    );
    return result;
}

inline Color operator*( real32 s, const Color& c )
{
    Color result(
        c.R * s,
        c.G * s,
        c.B * s
    );
    return result;
}

inline Color& operator*=( Color& c, real32 s )
{
    c.R *= s;
    c.G *= s;
    c.B *= s;
    return c;
}

inline Color& operator*=( real32 s, Color& c )
{
    c.R *= s;
    c.G *= s;
    c.B *= s;
    return c;
}

REX_NS_END

#endif