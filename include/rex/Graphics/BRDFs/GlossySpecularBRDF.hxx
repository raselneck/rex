#pragma once

#include "BRDF.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a glossy specular BRDF.
/// </summary>
class GlossySpecularBRDF : public BRDF
{
    REX_IMPLEMENT_DEVICE_MEM_OPS()

    Color  _color;
    real_t _coefficient;
    real_t _power;

public:
    /// <summary>
    /// Creates a new glossy-specular BRDF.
    /// </summary>
    __device__ GlossySpecularBRDF();

    /// <summary>
    /// Creates a new glossy-specular BRDF.
    /// </summary>
    /// <param name="ks">The specular coefficient.</param>
    /// <param name="color">The specular color.</param>
    /// <param name="pow">The specular power.</param>
    __device__ GlossySpecularBRDF( real_t ks, const Color& color, real_t pow );

    /// <summary>
    /// Destroys this glossy-specular BRDF.
    /// </summary>
    __device__ virtual ~GlossySpecularBRDF();

    /// <summary>
    /// Gets the bi-hemispherical reflectance. (rho in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    __device__ virtual Color GetBHR( const ShadePoint& sp, const Vector3& wo ) const;

    /// <summary>
    /// Gets the BRDF itself. (f in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The incoming light direction.</param>
    __device__ virtual Color GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const;

    /// <summary>
    /// Gets the specular coefficient.
    /// </summary>
    __device__ real_t GetSpecularCoefficient() const;

    /// <summary>
    /// Gets the specular color.
    /// </summary>
    __device__ const Color& GetSpecularColor() const;

    /// <summary>
    /// Gets the specular power.
    /// </summary>
    __device__ real_t GetSpecularPower() const;

    /// <summary>
    /// Sets the specular coefficient.
    /// </summary>
    /// <param name="ks">The new coefficient.</param>
    __device__ void SetSpecularCoefficient( real_t ks );

    /// <summary>
    /// Sets the specular color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __device__ void SetSpecularColor( const Color& color );

    /// <summary>
    /// Sets the specular color.
    /// </summary>
    /// <param name="r">The new color's red component..</param>
    /// <param name="g">The new color's green component..</param>
    /// <param name="b">The new color's blue component..</param>
    __device__ void SetSpecularColor( real_t r, real_t g, real_t b );

    /// <summary>
    /// Sets the specular power.
    /// </summary>
    /// <param name="pow">The new power.</param>
    __device__ void SetSpecularPower( real_t pow );
};

REX_NS_END