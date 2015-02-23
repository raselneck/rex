#ifndef __REX_COLOR_HXX
#define __REX_COLOR_HXX

#include "../Config.hxx"
#include "Math.hxx"

REX_NS_BEGIN

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

    /// <summary>
    /// Raises the given color to the given power.
    /// </summary>
    /// <param name="color">The color.</param>
    /// <param name="exp">The exponent.</param>
    static Color Pow( const Color& color, real32 exp );

    static const Color Black;
    static const Color White;
    static const Color Red;
    static const Color Green;
    static const Color Blue;
    static const Color Yellow;
    static const Color Cyan;
    static const Color Magenta;

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

#include "Color.inl"
#endif