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
    friend class Geometry;

    LambertianBRDF _ambient;
    LambertianBRDF _diffuse;

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    /// <param name="allocOnDevice">True to allocate this on the device, false to not.</param>
    __host__ MatteMaterial( const Color& color, real32 ka, real32 kd, bool allocOnDevice );

    /// <summary>
    /// Copies this material for geometry.
    /// </summary>
    __host__ virtual Material* Copy() const;

public:
    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    __host__ MatteMaterial();

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    __host__ MatteMaterial( const Color& color );

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    __host__ MatteMaterial( const Color& color, real32 ka, real32 kd );

    /// <summary>
    /// Destroys this matte material.
    /// </summary>
    __host__ virtual ~MatteMaterial();

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
    /// Gets this material on the device.
    /// </summary>
    __host__ virtual const Material* GetOnDevice() const;

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
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    /// <param name="lights">All of the lights in the scene.</param>
    /// <param name="lightCount">The number of lights in the scene</param>
    __device__ virtual Color Shade( ShadePoint& sp, const Light** lights, uint32 lightCount ) const;
};

REX_NS_END