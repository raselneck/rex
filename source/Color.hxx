#ifndef __REX_COLOR_HXX
#define __REX_COLOR_HXX
#pragma once

#include "Config.hxx"

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
    /// <param name="all"></param>
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
};

REX_NS_END

#endif