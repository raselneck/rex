#pragma once

#include "../Config.hxx"
#include <iostream>

REX_NS_BEGIN

/// <summary>
/// Defines a color.
/// </summary>
struct Color
{
    /// <summary>
    /// This color's red component.
    /// </summary>
    real_t R;

    /// <summary>
    /// This color's green component.
    /// </summary>
    real_t G;

    /// <summary>
    /// This color's blue component.
    /// </summary>
    real_t B;

    /// <summary>
    /// Creates a new color.
    /// </summary>
    __both__ Color();

    /// <summary>
    /// Creates a new color.
    /// </summary>
    /// <param name="all">The value to use for all components.</param>
    __both__ Color( real_t all );

    /// <summary>
    /// Creates a new color.
    /// </summary>
    /// <param name="r">The color's red component.</param>
    /// <param name="g">The color's green component.</param>
    /// <param name="b">The color's blue component.</param>
    __both__ Color( real_t r, real_t g, real_t b );

    /// <summary>
    /// Destroys this color.
    /// </summary>
    __both__ ~Color();

    /// <summary>
    /// Linearly interpolates two colors.
    /// </summary>
    /// <param name="c1">The first color.</param>
    /// <param name="c2">The second color.</param>
    /// <param name="amount">The amount to lerp the colors.</param>
    __both__ static Color Lerp( const Color& c1, const Color& c2, real_t amount );

    /// <summary>
    /// Darkens the given color.
    /// </summary>
    /// <param name="color">The color.</param>
    /// <param name="amount">The amount to darken the color.</param>
    __both__ static Color Darken( const Color& color, real_t amount );

    /// <summary>
    /// Lighten the given color.
    /// </summary>
    /// <param name="color">The color.</param>
    /// <param name="amount">The amount to lighten the color.</param>
    __both__ static Color Lighten( const Color& color, real_t amount );

    /// <summary>
    /// Raises the given color to the given power.
    /// </summary>
    /// <param name="color">The color.</param>
    /// <param name="exp">The exponent.</param>
    __both__ static Color Pow( const Color& color, real_t exp );

#pragma region Pre-defined Colors

    // NOTE : These are functions because CUDA does not allow static member variables :(

    /// <summary>
    /// Gets the color red.
    /// </summary>
    __both__ static Color Red();

    /// <summary>
    /// Gets the color blue.
    /// </summary>
    __both__ static Color Blue();

    /// <summary>
    /// Gets the color green.
    /// </summary>
    __both__ static Color Green();

    /// <summary>
    /// Gets the color magenta.
    /// </summary>
    __both__ static Color Magenta();

    /// <summary>
    /// Gets the color yellow.
    /// </summary>
    __both__ static Color Yellow();

    /// <summary>
    /// Gets the color cyan.
    /// </summary>
    __both__ static Color Cyan();

    /// <summary>
    /// Gets the color orange.
    /// </summary>
    __both__ static Color Orange();

    /// <summary>
    /// Gets the color purple.
    /// </summary>
    __both__ static Color Purple();

    /// <summary>
    /// Gets the color white.
    /// </summary>
    __both__ static Color White();

    /// <summary>
    /// Gets the color black.
    /// </summary>
    __both__ static Color Black();

#pragma endregion

#pragma region Operators

    __both__ bool operator==( const Color& c ) const;
    __both__ bool operator!=( const Color& c ) const;

    __both__ Color operator+( const Color& c ) const;
    __both__ Color operator-( const Color& c ) const;
    __both__ Color operator/( real_t s ) const;

    __both__ Color& operator+=( const Color& c );
    __both__ Color& operator-=( const Color& c );
    __both__ Color& operator*=( real_t s );
    __both__ Color& operator/=( real_t s );

#pragma endregion

};

__both__ Color operator*( const Color& c1, const Color& c2 );
__both__ Color operator*( const Color& c, real_t s );
__both__ Color operator*( real_t s, const Color& c );

__host__ std::ostream& operator<<( std::ostream& stream, const Color& color );

REX_NS_END