#ifndef __REX_PHONGMATERIAL_HXX
#define __REX_PHONGMATERIAL_HXX

#include "../Config.hxx"
#include "../BRDFs/GlossySpecularBRDF.hxx"
#include "../BRDFs/LambertianBRDF.hxx"
#include "MatteMaterial.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a Phong material.
/// </summary>
class PhongMaterial : public MatteMaterial
{
protected:
    Handle<GlossySpecularBRDF> _specular;

public:
    /// <summary>
    /// Creates a new Phong material.
    /// </summary>
    PhongMaterial();

    /// <summary>
    /// Creates a new Phong material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    PhongMaterial( const Color& color );

    /// <summary>
    /// Creates a new Phong material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    /// <param name="ks">The initial specular coefficient.</param>
    /// <param name="pow">The initial specular power.</param>
    PhongMaterial( const Color& color, real32 ka, real32 kd, real32 ks, real32 pow );

    /// <summary>
    /// Copies another Phong material.
    /// </summary>
    /// <param name="other">The other material to copy.</param>
    PhongMaterial( const PhongMaterial& other );

    /// <summary>
    /// Destroys this Phong material.
    /// </summary>
    virtual ~PhongMaterial();

    /// <summary>
    /// Gets the specular coefficient.
    /// </summary>
    real32 GetSpecularCoefficient() const;

    /// <summary>
    /// Gets the specular power.
    /// </summary>
    real32 GetSpecularPower() const;
    
    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    virtual void SetColor( const Color& color );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="r">The new color's red component..</param>
    /// <param name="g">The new color's green component..</param>
    /// <param name="b">The new color's blue component..</param>
    virtual void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets the sampler to use with this material.
    /// </summary>
    /// <param name="sampler">The sampler to use.</param>
    virtual void SetSampler( Handle<Sampler>& sampler );

    /// <summary>
    /// Sets the specular coefficient.
    /// </summary>
    /// <param name="ks">The new coefficient.</param>
    void SetSpecularCoefficient( real32 ks );

    /// <summary>
    /// Sets the specular power.
    /// </summary>
    /// <param name="pow">The new power.</param>
    void SetSpecularPower( real32 pow );

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color Shade( ShadePoint& sp );
};

REX_NS_END

#endif