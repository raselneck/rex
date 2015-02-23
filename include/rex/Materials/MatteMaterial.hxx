#ifndef __REX_MATTEMATERIAL_HXX
#define __REX_MATTEMATERIAL_HXX

#include "../Config.hxx"
#include "../BRDFs/LambertianBRDF.hxx"
#include "Material.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a matte material.
/// </summary>
class MatteMaterial : public Material
{
    Handle<LambertianBRDF> _ambient;
    Handle<LambertianBRDF> _diffuse;

public:
    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    MatteMaterial();

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    MatteMaterial( const Color& color );

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient..</param>
    /// <param name="kd">The initial diffuse coefficient..</param>
    MatteMaterial( const Color& color, real32 ka, real32 kd );

    /// <summary>
    /// Destroys this matte material.
    /// </summary>
    virtual ~MatteMaterial();

    /// <summary>
    /// Gets the ambient BRDF's diffuse coefficient.
    /// </summary>
    real32 GetAmbientCoefficient() const;

    /// <summary>
    /// Gets this material's color.
    /// </summary>
    Color GetColor() const;

    /// <summary>
    /// Gets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    real32 GetDiffuseCoefficient() const;

    /// <summary>
    /// Sets the ambient BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="ka">The new ambient coefficient.</param>
    void SetAmbientCoefficient( real32 ka );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    void SetColor( const Color& color );

    /// <summary>
    /// Sets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="kd">The new diffuse coefficient.</param>
    void SetDiffuseCoefficient( real32 kd );

    /// <summary>
    /// Sets the sampler to use with this material.
    /// </summary>
    /// <param name="sampler">The sampler to use.</param>
    void SetSampler( Handle<Sampler>& sampler );

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    virtual Color Shade( ShadePoint& sp );
};

REX_NS_END

#endif