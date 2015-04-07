#include <rex/Graphics/BRDFs/LambertianBRDF.hxx>
#include <rex/Math/Math.hxx>

REX_NS_BEGIN

// create Lambertian BRDF
LambertianBRDF::LambertianBRDF()
    : _kd( 0.0f ), _dc( Color::Black() )
{
}

// create Lambertian BRDF w/ coefficient, color
LambertianBRDF::LambertianBRDF( real32 kd, const Color& dc )
    : _kd( kd ), _dc( dc )
{
}

// destroy Lambertian BRDF
LambertianBRDF::~LambertianBRDF()
{
    _kd = 0.0f;
}

// get bi-hemispherical reflectance
__device__ Color LambertianBRDF::GetBHR( const ShadePoint& sp, const Vector3& wo ) const
{
    return _kd * _dc;
}

// get BRDF
__device__ Color LambertianBRDF::GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const
{
    return _kd * _dc * static_cast<real32>( Math::InvPi() );
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