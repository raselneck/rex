#pragma once

#include "Light.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a directional light.
/// </summary>
class DirectionalLight : public Light
{
    vec3    _direction;
    Color   _color;
    real32  _radianceScale;

public:
    /// <summary>
    /// Creates a new directional light.
    /// </summary>
    __device__ DirectionalLight();

    /// <summary>
    /// Creates a new directional light.
    /// </summary>
    /// <param name="direction">The light's direction.</param>
    __device__ DirectionalLight( const vec3& direction );

    /// <summary>
    /// Creates a new directional light.
    /// </summary>
    /// <param name="x">The light's X direction.</param>
    /// <param name="y">The light's Y direction.</param>
    /// <param name="z">The light's Z direction.</param>
    __device__ DirectionalLight( real32 x, real32 y, real32 z );

    /// <summary>
    /// Destroys this directional light.
    /// </summary>
    __device__ virtual ~DirectionalLight();

    /// <summary>
    /// Gets this light's color.
    /// </summary>
    __device__ const Color& GetColor() const;

    /// <summary>
    /// Gets this light's direction.
    /// </summary>
    __device__ const vec3& GetDirection() const;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual vec3 GetLightDirection( ShadePoint& sp ) const;

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
    /// Sets this light's direction.
    /// </summary>
    /// <param name="direction">The new direction.</param>
    __device__ void SetDirection( const vec3& direction );

    /// <summary>
    /// Sets this light's direction.
    /// </summary>
    /// <param name="x">The new direction's X component.</param>
    /// <param name="y">The new direction's Y component.</param>
    /// <param name="z">The new direction's Z component.</param>
    __device__ void SetDirection( real32 x, real32 y, real32 z );

    /// <summary>
    /// Sets this light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    __device__ void SetRadianceScale( real32 ls );
};

REX_NS_END