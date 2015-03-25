#pragma once

#include "Light.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an ambient light.
/// </summary>
class AmbientLight : public Light
{
    Color  _color;
    real32 _radianceScale;

public:
    /// <summary>
    /// Creates a new ambient light.
    /// </summary>
    __both__ AmbientLight();

    /// <summary>
    /// Destroys this ambient light.
    /// </summary>
    __both__ virtual ~AmbientLight();

    /// <summary>
    /// Gets this ambient light's color.
    /// </summary>
    __both__ const Color& GetColor() const;

    /// <summary>
    /// Gets this ambient light's radiance scale.
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
    __both__ virtual bool IsInShadow( const Ray& ray, const ShadePoint& sp ) const;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __both__ virtual Vector3 GetLightDirection( ShadePoint& sp );

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    __both__ virtual Color GetRadiance( ShadePoint& sp );

    /// <summary>
    /// Sets whether or not this light should cast shadows.
    /// </summary>
    /// <param name="value">The new value.</param>
    __both__ virtual void SetCastShadows( bool value );

    /// <summary>
    /// Sets this ambient light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __both__ void SetColor( const Color& color );

    /// <summary>
    /// Sets this ambient light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    __both__ void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this ambient light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    __both__ void SetRadianceScale( real32 ls );
};

REX_NS_END