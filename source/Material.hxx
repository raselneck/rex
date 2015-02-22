#ifndef __REX_MATERIAL_HXX
#define __REX_MATERIAL_HXX

#include "Config.hxx"
#include "Color.hxx"

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
    Material();

    /// <summary>
    /// Destroys this material.
    /// </summary>
    virtual ~Material();

    /// <summary>
    /// Gets an area-light-shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color AreaLightShade( ShadePoint& sp );

    /// <summary>
    /// Gets a path-shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color PathShade( ShadePoint& sp );

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color Shade( ShadePoint& sp );

    /// <summary>
    /// Gets a Whitted-shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color WhittedShade( ShadePoint& sp );
};

REX_NS_END

#endif