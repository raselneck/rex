#ifndef __REX_MATERIAL_HXX
#define __REX_MATERIAL_HXX

#include "../Config.hxx"
#include "../Utility/Color.hxx"

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
    /// Copies this material.
    /// </summary>
    virtual Handle<Material> Copy() const = 0;

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color Shade( ShadePoint& sp );
};

REX_NS_END

#endif