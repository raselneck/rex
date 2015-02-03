#ifndef __REX_COLOR_HXX
#define __REX_COLOR_HXX
#pragma once

#include "Config.hxx"

REX_NS_BEGIN

struct Vector3;

/// <summary>
/// Defines a color.
/// </summary>
struct Color
{
    real32 R;
    real32 G;
    real32 B;

    /// <summary>
    /// Creates a new color.
    /// </summary>
    Color();

    /// <summary>
    /// Creates a new color.
    /// </summary>
    /// <param name="all">The value to use for all components.</param>
    Color( real32 all );

    /// <summary>
    /// Creates a new color.
    /// </summary>
    /// <param name="r">The red value.</param>
    /// <param name="g">The green value.</param>
    /// <param name="b">The blue value.</param>
    Color( real32 r, real32 g, real32 b );

    /// <summary>
    /// Destroys this color.
    /// </summary>
    ~Color();

    /// <summary>
    /// Gets the average of all of the components in this color.
    /// </summary>
    real32 GetAverage() const;

    operator Vector3();

    bool operator==( const Color& ) const;
    bool operator!=( const Color& ) const;

    Color operator+( const Color& ) const;
    Color operator-( const Color& ) const;
    Color operator*( const Color& ) const;
    Color operator/( const Color& ) const;

    Color& operator+=( const Color& );
    Color& operator-=( const Color& );
    Color& operator*=( const Color& );
    Color& operator/=( const Color& );
};

Color operator*( const Color&, real32 );
Color operator*( real32, const Color& );
Color& operator*=( Color&, real32 );
Color& operator*=( real32, Color& );

REX_NS_END

#endif