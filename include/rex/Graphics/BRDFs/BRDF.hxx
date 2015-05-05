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
    __device__ BRDF();

    /// <summary>
    /// Destroys this BRDF.
    /// </summary>
    __device__ virtual ~BRDF();

    /// <summary>
    /// Gets the BRDF itself. (f in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The incoming light direction.</param>
    __device__ virtual Color GetBRDF( const ShadePoint& sp, const vec3& wo, const vec3& wi ) const = 0;

    /// <summary>
    /// Gets the bi-hemispherical reflectance. (rho in Suffern.)
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    __device__ virtual Color GetBHR( const ShadePoint& sr, const vec3& wo ) const = 0;

    /// <summary>
    /// Samples the BRDF.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The (calculated) incoming light direction.</param>
    /// __device__ Color Sample( const ShadePoint& sp, vec3& wo, const vec3& wi ) const = delete;
};

REX_NS_END