#pragma once

#include "../../Config.hxx"
#include "../Color.hxx"

REX_NS_BEGIN

struct ShadePoint;

/// <summary>
/// Defines the base for all materials.
/// </summary>
class Material
{
public:
    /// <summary>
    /// Creates a new material.
    /// </summary>
    __both__ Material();

    /// <summary>
    /// Destroys this material.
    /// </summary>
    __both__ virtual ~Material();

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    __both__ virtual Color Shade( ShadePoint& sp );
};

REX_NS_END