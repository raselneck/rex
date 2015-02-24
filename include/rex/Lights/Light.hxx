#ifndef __REX_LIGHT_HXX
#define __REX_LIGHT_HXX

#include "../Config.hxx"
#include "../Utility/Color.hxx"
#include "../Utility/Ray.hxx"
#include "../Utility/Vector3.hxx"

REX_NS_BEGIN

struct ShadePoint;

/// <summary>
/// Defines the base for lights.
/// </summary>
class Light
{
protected:
    bool _castShadows;

public:
    /// <summary>
    /// Creates a new light.
    /// </summary>
    Light();

    /// <summary>
    /// Destroys this light.
    /// </summary>
    virtual ~Light();

    /// <summary>
    /// Checks to see if this light casts shadows.
    /// </summary>
    bool CastsShadows() const;

    /// <summary>
    /// Checks to see if the given ray is in shadow when viewed from this light.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="sp">Current hit point information.</param>
    virtual bool IsInShadow( const Ray& ray, const ShadePoint& sp ) const = 0;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    virtual Vector3 GetLightDirection( ShadePoint& sp ) = 0;

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    virtual Color GetRadiance( ShadePoint& sp ) = 0;

    /// <summary>
    /// Sets whether or not this light should cast shadows.
    /// </summary>
    /// <param name="value">The new value.</param>
    virtual void SetCastShadows( bool value );
};

REX_NS_END

#endif