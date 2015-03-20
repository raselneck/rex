#pragma once

#include "../../Config.hxx"
#include "../../Math/Ray.hxx"
#include "../../Math/Vector3.hxx"
#include "../Color.hxx"

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
    __both__ Light();

    /// <summary>
    /// Destroys this light.
    /// </summary>
    __both__ virtual ~Light();

    /// <summary>
    /// Checks to see if this light casts shadows.
    /// </summary>
    __both__ bool CastsShadows() const;

    /// <summary>
    /// Checks to see if the given ray is in shadow when viewed from this light.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="sp">Current hit point information.</param>
    __both__ virtual bool IsInShadow( const Ray& ray, const ShadePoint& sp ) const = 0;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __both__ virtual Vector3 GetLightDirection( ShadePoint& sp ) = 0;

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __both__ virtual Color GetRadiance( ShadePoint& sp ) = 0;

    /// <summary>
    /// Sets whether or not this light should cast shadows.
    /// </summary>
    /// <param name="value">The new value.</param>
    __both__ virtual void SetCastShadows( bool value );
};

REX_NS_END