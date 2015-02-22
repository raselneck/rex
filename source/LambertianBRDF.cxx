#include "LambertianBRDF.hxx"
#include "Math.hxx"

REX_NS_BEGIN

// create Lambertian BRDF
LambertianBRDF::LambertianBRDF()
    : _kd( 0.0f ), _dc( Color::Black )
{
}

// create Lambertian BRDF w/ coefficient, color
LambertianBRDF::LambertianBRDF( real32 kd, const Color& dc )
    : _kd( kd ), _dc( dc )
{
}

// create Lambertian BRDF w/ coefficient, color, sampler
LambertianBRDF::LambertianBRDF( real32 kd, const Color& dc, Handle<Sampler>& sampler )
    : BRDF( sampler ), _kd( kd ), _dc( dc )
{
}

// destroy Lambertian BRDF
LambertianBRDF::~LambertianBRDF()
{
    _kd = 0.0f;
}

// get bi-hemispherical reflectance
Color LambertianBRDF::GetBHR( const ShadePoint& sp, const Vector3& wo ) const
{
    return _kd * _dc;
}

// get BRDF
Color LambertianBRDF::GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const
{
    return _kd * _dc * real32( Math::INV_PI );
}

// get diffuse color
Color LambertianBRDF::GetDiffuseColor() const
{
    return _dc;
}

// get diffuse reflection coefficient
real32 LambertianBRDF::GetDiffuseCoefficient() const
{
    return _kd;
}

// sample the BRDF
Color LambertianBRDF::Sample( const ShadePoint& sp, Vector3& wo, const Vector3& wi ) const
{
    // TODO : Temporary?
    return Color::Black;
}

// set diffuse color
void LambertianBRDF::SetDiffuseColor( const Color& color )
{
    _dc = color;
}

// set diffuse reflection coefficient
void LambertianBRDF::SetDiffuseCoefficient( real32 coeff )
{
    _kd = coeff;
}

REX_NS_END