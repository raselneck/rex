#pragma once

#include "Light.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a point light.
/// </summary>
class PointLight : public Light
{
    Vector3 _position;
    Color   _color;
    real32  _radianceScale;
    void*   _dThis;
protected:
    friend class Scene;

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
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __device__ virtual Color GetRadiance( ShadePoint& sp );

public:
    /// <summary>
    /// Creates a new point light.
    /// </summary>
    __host__ PointLight();

    /// <summary>
    /// Creates a new point light.
    /// </summary>
    /// <param name="position">The light's coordinates.</param>
    __host__ PointLight( const Vector3& position );

    /// <summary>
    /// Creates a new point light.
    /// </summary>
    /// <param name="x">The light's X coordinate.</param>
    /// <param name="y">The light's Y coordinate.</param>
    /// <param name="z">The light's Z coordinate.</param>
    __host__ PointLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Destroys this point light.
    /// </summary>
    __host__ virtual ~PointLight();

    /// <summary>
    /// Gets this light's color.
    /// </summary>
    __both__ const Color& GetColor() const;

    /// <summary>
    /// Gets this light on the device.
    /// </summary>
    __host__ virtual const Light* GetOnDevice() const;

    /// <summary>
    /// Gets this light's position.
    /// </summary>
    __both__ const Vector3& GetPosition() const;

    /// <summary>
    /// Gets this light's radiance scale.
    /// </summary>
    __both__ real32 GetRadianceScale() const;

    /// <summary>
    /// Gets this light's type.
    /// </summary>
    __both__ virtual LightType GetType() const;

    /// <summary>
    /// Sets this light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __host__ void SetColor( const Color& color );

    /// <summary>
    /// Sets this light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    __host__ void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this light's position.
    /// </summary>
    /// <param name="position">The new position.</param>
    __host__ void SetPosition( const Vector3& position );

    /// <summary>
    /// Sets this light's position.
    /// </summary>
    /// <param name="x">The new position's X coordinate.</param>
    /// <param name="y">The new position's Y coordinate.</param>
    /// <param name="z">The new position's Z coordinate.</param>
    __host__ void SetPosition( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets this light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    __host__ void SetRadianceScale( real32 ls );
};

REX_NS_END