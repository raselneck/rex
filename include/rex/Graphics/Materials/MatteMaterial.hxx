#pragma once

#include "Material.hxx"
#include "../BRDFs/LambertianBRDF.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a matte material.
/// </summary>
class MatteMaterial : public Material
{
protected:
    LambertianBRDF _ambient;
    LambertianBRDF _diffuse;

public:
    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    __both__ MatteMaterial();

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    __both__ MatteMaterial( const Color& color );

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    __both__ MatteMaterial( const Color& color, real32 ka, real32 kd );

    /// <summary>
    /// Destroys this matte material.
    /// </summary>
    __both__ virtual ~MatteMaterial();

    /// <summary>
    /// Gets the ambient BRDF's diffuse coefficient.
    /// </summary>
    __both__ real32 GetAmbientCoefficient() const;

    /// <summary>
    /// Gets this material's color.
    /// </summary>
    __both__ Color GetColor() const;

    /// <summary>
    /// Gets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    __both__ real32 GetDiffuseCoefficient() const;

    /// <summary>
    /// Sets the ambient BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="ka">The new ambient coefficient.</param>
    __both__ void SetAmbientCoefficient( real32 ka );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __both__ virtual void SetColor( const Color& color );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="r">The new color's red component..</param>
    /// <param name="g">The new color's green component..</param>
    /// <param name="b">The new color's blue component..</param>
    __both__ virtual void SetColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="kd">The new diffuse coefficient.</param>
    __both__ void SetDiffuseCoefficient( real32 kd );

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    __both__ virtual Color Shade( ShadePoint& sp );
};

REX_NS_END