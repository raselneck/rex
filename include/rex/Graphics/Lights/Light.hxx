#pragma once

#include "../../Math/Ray.hxx"
#include "../../Math/Vector3.hxx"
#include "../Color.hxx"

REX_NS_BEGIN

struct ShadePoint;

/// <summary>
/// An enumeration of possible light types.
/// </summary>
enum class LightType
{
    AmbientLight,
    DirectionalLight,
    PointLight
};

/// <summary>
/// Defines the base for lights.
/// </summary>
class Light
{
    REX_NONCOPYABLE_CLASS( Light );

    friend class Scene;

protected:
    bool  _castShadows;
    void* _dThis;

public:
    /// <summary>
    /// Creates a new light.
    /// </summary>
    __host__ Light();

    /// <summary>
    /// Destroys this light.
    /// </summary>
    __host__ virtual ~Light();

    /// <summary>
    /// Checks to see if this light casts shadows.
    /// </summary>
    __both__ bool CastsShadows() const;

    /// <summary>
    /// Gets this light on the device.
    /// </summary>
    __host__ virtual const Light* GetOnDevice() const = 0;

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
    __both__ virtual LightType GetType() const = 0;

    /// <summary>
    /// Checks to see if the given ray is in shadow when viewed from this light.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="sp">Current hit point information.</param>
    __device__ virtual bool IsInShadow( const Ray& ray, const ShadePoint& sp ) const = 0;

    /// <summary>
    /// Sets whether or not this light should cast shadows.
    /// </summary>
    /// <param name="value">The new value.</param>
    __host__ virtual void SetCastShadows( bool value );
};

REX_NS_END