#ifndef __REX_POINTLIGHT_HXX
#define __REX_POINTLIGHT_HXX

#include "Config.hxx"
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

public:
    /// <summary>
    /// Creates a new point light.
    /// </summary>
    PointLight();

    /// <summary>
    /// Creates a new point light.
    /// </summary>
    /// <param name="position">The point light's coordinates.</param>
    PointLight( const Vector3& position );

    /// <summary>
    /// Creates a new point light.
    /// </summary>
    /// <param name="x">The point light's X coordinate.</param>
    /// <param name="y">The point light's Y coordinate.</param>
    /// <param name="z">The point light's Z coordinate.</param>
    PointLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Destroys this point light.
    /// </summary>
    virtual ~PointLight();

    /// <summary>
    /// Gets this point light's color.
    /// </summary>
    const Color& GetColor() const;

    /// <summary>
    /// Gets this point light's position.
    /// </summary>
    const Vector3& GetPosition() const;

    /// <summary>
    /// Gets this point light's radiance scale.
    /// </summary>
    real32 GetRadianceScale() const;

    /// <summary>
    /// Gets the direction of the incoming light at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    virtual Vector3 GetLightDirection( ShadePoint& sp );

    /// <summary>
    /// Gets the incident radiance at a hit point.
    /// </summary>
    /// <param name="sp">The shading point information containing hit data.</param>
    virtual Color GetRadiance( ShadePoint& sp );

    /// <summary>
    /// Sets this point light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    void SetColor( const Color& color );

    /// <summary>
    /// Sets this point light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this point light's position.
    /// </summary>
    /// <param name="position">The new position.</param>
    void SetPosition( const Vector3& position );

    /// <summary>
    /// Sets this point light's position.
    /// </summary>
    /// <param name="x">The new position's X coordinate.</param>
    /// <param name="y">The new position's Y coordinate.</param>
    /// <param name="z">The new position's Z coordinate.</param>
    void SetPosition( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets this point light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    void SetRadianceScale( real32 ls );
};

REX_NS_END

#endif