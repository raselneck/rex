#ifndef __REX_BRDF_HXX
#define __REX_BRDF_HXX

#include "Config.hxx"
#include "Color.hxx"
#include "Sampler.hxx"
#include "ShadePoint.hxx"

/**
 * Notes about BRDFs:
 * 1) Irradiance can be thought of as the light received.
 * 2) Radiance can be thought of as the light given off.
 */

REX_NS_BEGIN

/// <summary>
/// Defines the basis for all bidirectional reflectance distribution functions.
/// </summary>
class BRDF
{
protected:
    Handle<Sampler> _sampler;

public:
    /// <summary>
    /// Creates a new BRDF.
    /// </summary>
    BRDF();

    /// <summary>
    /// Creates a new BRDF.
    /// </summary>
    /// <param name="sampler">The sampler to use with this BRDF.</param>
    BRDF( Handle<Sampler>& sampler );

    /// <summary>
    /// Destroys this BRDF.
    /// </summary>
    virtual ~BRDF();

    /// <summary>
    /// Gets the BRDF itself.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The incoming light direction.</param>
    virtual Color GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const = 0;

    /// <summary>
    /// Samples the BRDF.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    /// <param name="wi">The (calculated) incoming light direction.</param>
    virtual Color Sample( const ShadePoint& sp, Vector3& wo, const Vector3& wi ) const = 0;

    /// <summary>
    /// Gets the bi-hemispherical reflectance.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    virtual Color GetBHR( const ShadePoint& sr, const Vector3& wo ) const = 0;

    /// <summary>
    /// Sets the sampler used by this BRDF.
    /// </summary>
    /// <param name="sampler">The new sampler.</param>
    void SetSampler( Handle<Sampler>& sampler );
};

REX_NS_END

#endif