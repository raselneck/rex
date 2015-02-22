#ifndef __REX_AMBIENTLIGHT_HXX
#define __REX_AMBIENTLIGHT_HXX

#include "Config.hxx"
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
    AmbientLight();

    /// <summary>
    /// Destroys this ambient light.
    /// </summary>
    virtual ~AmbientLight();

    /// <summary>
    /// Gets this ambient light's color.
    /// </summary>
    const Color& GetColor() const;

    /// <summary>
    /// Gets this ambient light's radiance scale.
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
    /// Sets this ambient light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    void SetColor( const Color& color );

    /// <summary>
    /// Sets this ambient light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this ambient light's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    void SetRadianceScale( real32 ls );
};

REX_NS_END

#endif