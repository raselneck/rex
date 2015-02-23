#ifndef __REX_GLOSSYSPECULARBRDF_HXX
#define __REX_GLOSSYSPECULARBRDF_HXX

#include "../Config.hxx"
#include "../Utility/Color.hxx"
#include "BRDF.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a glossy specular BRDF.
/// </summary>
class GlossySpecularBRDF : public BRDF
{
    Color  _color;
    real32 _ks;
    real32 _pow;

public:
    /// <summary>
    /// Creates a new glossy-specular BRDF.
    /// </summary>
    GlossySpecularBRDF();

    /// <summary>
    /// Creates a new glossy-specular BRDF.
    /// </summary>
    /// <param name="ks">The specular coefficient.</param>
    /// <param name="color">The specular color.</param>
    /// <param name="pow">The specular power.</param>
    GlossySpecularBRDF( real32 ks, const Color& color, real32 pow );

    /// <summary>
    /// Creates a new glossy-specular BRDF.
    /// </summary>
    /// <param name="ks">The specular coefficient.</param>
    /// <param name="color">The specular color.</param>
    /// <param name="pow">The specular power.</param>
    /// <param name="sampler">The sampler to use with this BRDF.</param>
    GlossySpecularBRDF( real32 ks, const Color& color, real32 pow, Handle<Sampler>& sampler );

    /// <summary>
    /// Destroys this glossy-specular BRDF.
    /// </summary>
    virtual ~GlossySpecularBRDF();

    /// <summary>
    /// Gets the bi-hemispherical reflectance. (rho in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    virtual Color GetBHR( const ShadePoint& sp, const Vector3& wo ) const;

    /// <summary>
    /// Gets the BRDF itself. (f in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The incoming light direction.</param>
    virtual Color GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const;

    /// <summary>
    /// Gets the specular coefficient.
    /// </summary>
    real32 GetSpecularCoefficient() const;

    /// <summary>
    /// Gets the specular color.
    /// </summary>
    const Color& GetSpecularColor() const;

    /// <summary>
    /// Gets the specular power.
    /// </summary>
    real32 GetSpecularPower() const;

    /// <summary>
    /// Samples the BRDF. (sample_f in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The (calculated) incoming light direction.</param>
    virtual Color Sample( const ShadePoint& sp, Vector3& wo, const Vector3& wi ) const;

    /// <summary>
    /// Sets the specular coefficient.
    /// </summary>
    /// <param name="ks">The new coefficient.</param>
    void SetSpecularCoefficient( real32 ks );

    /// <summary>
    /// Sets the specular color.
    /// </summary>
    /// <param name="color">The new color.</param>
    void SetSpecularColor( const Color& color );

    /// <summary>
    /// Sets the specular color.
    /// </summary>
    /// <param name="r">The new color's red component..</param>
    /// <param name="g">The new color's green component..</param>
    /// <param name="b">The new color's blue component..</param>
    void SetSpecularColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets the specular power.
    /// </summary>
    /// <param name="pow">The new power.</param>
    void SetSpecularPower( real32 pow );
};

REX_NS_END

#endif