#pragma once

#include "Light.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a directional light.
/// </summary>
class DirectionalLight : public Light
{
    Vector3 _direction;
    Color   _color;
    real32  _radianceScale;
    void*   _dThis;

public:
    /// <summary>
    /// Creates a new directional light.
    /// </summary>
    __both__ DirectionalLight();

    /// <summary>
    /// Creates a new directional light.
    /// </summary>
    /// <param name="direction">The light's direction.</param>
    __both__ DirectionalLight( const Vector3& direction );

    /// <summary>
    /// Creates a new directional light.
    /// </summary>
    /// <param name="x">The light's X direction.</param>
    /// <param name="y">The light's Y direction.</param>
    /// <param name="z">The light's Z direction.</param>
    __both__ DirectionalLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Destroys this directional light.
    /// </summary>
    __both__ virtual ~DirectionalLight();

    /// <summary>
    /// Gets this light's color.
    /// </summary>
    __both__ const Color& GetColor() const;

    /// <summary>
    /// Gets this light's direction.
    /// </summary>
    __both__ const Vector3& GetDirection() const;

    /// <summary>
    /// Gets this light's radiance scale.
    /// </summary>
    __both__ real32 GetRadianceScale() const;

    /// <summary>
    /// Gets this light's type.
    /// </summary>
    __both__ virtual LightType GetType() const;

    /// <summary>
    /// Checks to see if the given ray is in shadow when viewed from this light.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="sp">Current hit point information.</param>
    __device__ virtual bool IsInShadow( const Ray& ray, const ShadePoint& sp ) const;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual Vector3 GetLightDirection( ShadePoint& sp );

    /// <summary>
    /// Gets this light on the device.
    /// </summary>
    __host__ virtual Light* GetOnDevice();

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual Color GetRadiance( ShadePoint& sp );

    /// <summary>
    /// Sets this light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __both__ void SetColor( const Color& color );

    /// <summary>
    /// Sets this light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    __both__ void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this light's direction.
    /// </summary>
    /// <param name="direction">The new direction.</param>
    __both__ void SetDirection( const Vector3& direction );

    /// <summary>
    /// Sets this light's direction.
    /// </summary>
    /// <param name="x">The new direction's X component.</param>
    /// <param name="y">The new direction's Y component.</param>
    /// <param name="z">The new direction's Z component.</param>
    __both__ void SetDirection( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets this light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    __both__ void SetRadianceScale( real32 ls );
};

REX_NS_END