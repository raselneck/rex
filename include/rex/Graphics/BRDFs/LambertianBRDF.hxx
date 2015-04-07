#pragma once

#include "BRDF.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a Lambertian bidirectional reflectance distribution function.
/// </summary>
class LambertianBRDF : public BRDF
{
    real32 _kd;
    Color  _dc;

public:
    /// <summary>
    /// Creates a new Lambertian BRDF.
    /// </summary>
    __host__ LambertianBRDF();

    /// <summary>
    /// Creates a new Lambertian BRDF.
    /// </summary>
    /// <param name="kd">The diffuse reflection coefficient.</param>
    /// <param name="dc">The diffuse color.</param>
    __host__ LambertianBRDF( real32 kd, const Color& dc );

    /// <summary>
    /// Destroys this Lambertian BRDF.
    /// </summary>
    __host__ ~LambertianBRDF();

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
    /// Gets the diffuse color.
    /// </summary>
    __both__ Color GetDiffuseColor() const;

    /// <summary>
    /// Gets the diffuse reflection coefficient.
    /// </summary>
    __both__ real32 GetDiffuseCoefficient() const;

    /// <summary>
    /// Sets the diffuse color.
    /// </summary>
    /// <param name="color">The new diffuse color.</param>
    __host__ void SetDiffuseColor( const Color& color );

    /// <summary>
    /// Sets the diffuse reflection coefficient.
    /// </summary>
    /// <param name="coeff">The new diffuse reflection coefficient.</param>
    __host__ void SetDiffuseCoefficient( real32 coeff );
};

REX_NS_END