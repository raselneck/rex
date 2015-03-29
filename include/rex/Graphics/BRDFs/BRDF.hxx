#pragma once

#include "../../Config.hxx"
#include "../ShadePoint.hxx"
#include "../Color.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines the basis for all bidirectional reflectance distribution functions.
/// </summary>
class BRDF
{
public:
    /// <summary>
    /// Creates a new BRDF.
    /// </summary>
    __host__ BRDF();

    /// <summary>
    /// Destroys this BRDF.
    /// </summary>
    __host__ virtual ~BRDF();

    /// <summary>
    /// Gets the BRDF itself.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The incoming light direction.</param>
    __device__ virtual Color GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const = 0;

    /// <summary>
    /// Gets the bi-hemispherical reflectance.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    __device__ virtual Color GetBHR( const ShadePoint& sr, const Vector3& wo ) const = 0;

    /// <summary>
    /// Samples the BRDF.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The (calculated) incoming light direction.</param>
    /// __device__ Color Sample( const ShadePoint& sp, Vector3& wo, const Vector3& wi ) const = delete;
};

REX_NS_END