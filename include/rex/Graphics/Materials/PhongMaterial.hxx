#pragma once

#include "MatteMaterial.hxx"
#include "../BRDFs/GlossySpecularBRDF.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a Phong material.
/// </summary>
class PhongMaterial : public MatteMaterial
{
protected:
    GlossySpecularBRDF _specular;

public:
    /// <summary>
    /// Creates a new Phong material.
    /// </summary>
    __host__ PhongMaterial();

    /// <summary>
    /// Creates a new Phong material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    __host__ PhongMaterial( const Color& color );

    /// <summary>
    /// Creates a new Phong material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    /// <param name="ks">The initial specular coefficient.</param>
    /// <param name="pow">The initial specular power.</param>
    __host__ PhongMaterial( const Color& color, real32 ka, real32 kd, real32 ks, real32 pow );

    /// <summary>
    /// Destroys this Phong material.
    /// </summary>
    __host__ virtual ~PhongMaterial();

    /// <summary>
    /// Gets the specular coefficient.
    /// </summary>
    __both__ real32 GetSpecularCoefficient() const;

    /// <summary>
    /// Gets the specular power.
    /// </summary>
    __both__ real32 GetSpecularPower() const;

    /// <summary>
    /// Gets this material's type.
    /// </summary>
    __both__ virtual MaterialType GetType() const;

    /// <summary>
    /// Sets the ambient BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="ka">The new ambient coefficient.</param>
    __host__ virtual void SetAmbientCoefficient( real32 ka );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __host__ virtual void SetColor( const Color& color );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="r">The new color's red component..</param>
    /// <param name="g">The new color's green component..</param>
    /// <param name="b">The new color's blue component..</param>
    __host__ virtual void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="kd">The new diffuse coefficient.</param>
    __host__ virtual void SetDiffuseCoefficient( real32 kd );

    /// <summary>
    /// Sets the specular coefficient.
    /// </summary>
    /// <param name="ks">The new coefficient.</param>
    __host__ void SetSpecularCoefficient( real32 ks );

    /// <summary>
    /// Sets the specular power.
    /// </summary>
    /// <param name="pow">The new power.</param>
    __host__ void SetSpecularPower( real32 pow );

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    __device__ virtual Color Shade( ShadePoint& sp );
};

REX_NS_END