#pragma once

#include "../../Math/Ray.hxx"
#include "../../Math/Vector3.hxx"
#include "../Geometry/Octree.hxx"
#include "../Color.hxx"

REX_NS_BEGIN

struct ShadePoint;

/// <summary>
/// An enumeration of possible light types.
/// </summary>
enum class LightType
{
    Ambient,
    Directional,
    Point
};

/// <summary>
/// Defines the base for lights.
/// </summary>
class Light
{
    REX_NONCOPYABLE_CLASS( Light )
    REX_IMPLEMENT_DEVICE_MEM_OPS()

protected:
    bool  _castShadows;
    LightType _type;
    
public:
    /// <summary>
    /// Creates a new light.
    /// </summary>
    /// <param name="type">This light's type.</param>
    __device__ Light( LightType type );

    /// <summary>
    /// Destroys this light.
    /// </summary>
    __device__ virtual ~Light();

    /// <summary>
    /// Checks to see if this light casts shadows.
    /// </summary>
    __device__ bool CastsShadows() const;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual Vector3 GetLightDirection( ShadePoint& sp ) const = 0;

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual Color GetRadiance( ShadePoint& sp ) const = 0;

    /// <summary>
    /// Gets this light's type.
    /// </summary>
    __device__ LightType GetType() const;

    /// <summary>
    /// Checks to see if the given ray is in shadow when viewed from this light.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="octree">The octree containing all of the geometry to check for.</param>
    /// <param name="sp">Current hit point information.</param>
    __device__ virtual bool IsInShadow( const Ray& ray, const Octree* octree, const ShadePoint& sp ) const = 0;

    /// <summary>
    /// Sets whether or not this light should cast shadows.
    /// </summary>
    /// <param name="value">The new value.</param>
    __device__ virtual void SetCastShadows( bool value );
};

REX_NS_END