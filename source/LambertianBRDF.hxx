#ifndef __REX_LAMBERTIANBRDF_HXX
#define __REX_LAMBERTIANBRDF_HXX

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
    /// <param name="sampler">The sampler to use with this BRDF.</param>
    /// <param name="kd">The diffuse reflection coefficient.</param>
    /// <param name="dc">The diffuse color.</param>
    LambertianBRDF( Handle<Sampler>& sampler, real32 kd, const Color& dc );

    /// <summary>
    /// Destroys this Lambertian BRDF.
    /// </summary>
    ~LambertianBRDF();

    /// <summary>
    /// Gets the bi-hemispherical reflectance.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    virtual Color GetBHR( const ShadePoint& sr, const Vector3& wo ) const;

    /// <summary>
    /// Gets the BRDF itself.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wi">The incoming light direction.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    virtual Color GetBRDF( const ShadePoint& sp, const Vector3& wi, const Vector3& wo ) const;

    /// <summary>
    /// Gets the diffuse color.
    /// </summary>
    Color GetDiffuseColor() const;

    /// <summary>
    /// Gets the diffuse reflection coefficient.
    /// </summary>
    Color GetDiffuseCoefficient() const;

    /// <summary>
    /// Samples the BRDF.
    /// </summary>
    /// <param name="sp">The shade point information.</param>
    /// <param name="wi">The (calculated) incoming light direction.</param>
    /// <param name="wo">The outgoing, reflected light direction.</param>
    virtual Color Sample( const ShadePoint& sp, Vector3& wi, const Vector3& wo ) const;

    /// <summary>
    /// Sets the diffuse color.
    /// </summary>
    /// <param name="color">The new diffuse color.</param>
    void SetDiffuseColor( const Color& color );

    /// <summary>
    /// Sets the diffuse reflection coefficient.
    /// </summary>
    /// <param name="coeff">The new diffuse reflection coefficient.</param>
    void SetDiffuseCoefficient( real32 coeff );
};

REX_NS_END

#endif