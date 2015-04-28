#pragma once

#include "Light.hxx"
#include "../Geometry/Geometry.hxx"
#include "../../CUDA/DeviceList.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a point light.
/// </summary>
class PointLight : public Light
{
    vec3    _position;
    Color   _color;
    real32  _radianceScale;

public:
    /// <summary>
    /// Creates a new point light.
    /// </summary>
    __device__ PointLight();

    /// <summary>
    /// Creates a new point light.
    /// </summary>
    /// <param name="position">The light's coordinates.</param>
    __device__ PointLight( const vec3& position );

    /// <summary>
    /// Creates a new point light.
    /// </summary>
    /// <param name="x">The light's X coordinate.</param>
    /// <param name="y">The light's Y coordinate.</param>
    /// <param name="z">The light's Z coordinate.</param>
    __device__ PointLight( real32 x, real32 y, real32 z );

    /// <summary>
    /// Destroys this point light.
    /// </summary>
    __device__ virtual ~PointLight();

    /// <summary>
    /// Gets this light's color.
    /// </summary>
    __device__ const Color& GetColor() const;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual vec3 GetLightDirection( ShadePoint& sp ) const;

    /// <summary>
    /// Gets this light's position.
    /// </summary>
    __device__ const vec3& GetPosition() const;

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual Color GetRadiance( ShadePoint& sp ) const;

    /// <summary>
    /// Gets this light's radiance scale.
    /// </summary>
    __device__ real32 GetRadianceScale() const;

    /// <summary>
    /// Checks to see if the given ray is in shadow when viewed from this light.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="octree">The octree containing all of the geometry to check for.</param>
    /// <param name="sp">Current hit point information.</param>
    __device__ virtual bool IsInShadow( const Ray& ray, const Octree* octree, const ShadePoint& sp ) const;

    /// <summary>
    /// Sets this light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __device__ void SetColor( const Color& color );

    /// <summary>
    /// Sets this light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    __device__ void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this light's position.
    /// </summary>
    /// <param name="position">The new position.</param>
    __device__ void SetPosition( const vec3& position );

    /// <summary>
    /// Sets this light's position.
    /// </summary>
    /// <param name="x">The new position's X coordinate.</param>
    /// <param name="y">The new position's Y coordinate.</param>
    /// <param name="z">The new position's Z coordinate.</param>
    __device__ void SetPosition( real32 x, real32 y, real32 z );

    /// <summary>
    /// Sets this light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    __device__ void SetRadianceScale( real32 ls );
};

REX_NS_END