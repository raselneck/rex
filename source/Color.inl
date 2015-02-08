#ifndef __REX_COLOR_INL
#define __REX_COLOR_INL
#pragma once

#include "Color.hxx"

REX_NS_BEGIN

// get average
inline real32 Color::GetAverage() const
{
    return real32( ( R + G + B ) * Math::ONE_THIRD );
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