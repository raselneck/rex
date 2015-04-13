#include <rex/Graphics/BRDFs/LambertianBRDF.hxx>
#include <rex/Math/Math.hxx>

REX_NS_BEGIN

// create Lambertian BRDF
__device__ LambertianBRDF::LambertianBRDF()
    : LambertianBRDF( 0.0f, Color::Black() )
{
}

// create Lambertian BRDF w/ coefficient, color
__device__ LambertianBRDF::LambertianBRDF( real_t kd, const Color& dc )
    : _coefficient( kd ),
      _color( dc )
{
}

// destroy Lambertian BRDF
__device__ LambertianBRDF::~LambertianBRDF()
{
    _coefficient = 0.0f;
}

// get bi-hemispherical reflectance
__device__ Color LambertianBRDF::GetBHR( const ShadePoint& sp, const Vector3& wo ) const
{
    return _coefficient * _color;
}

// get BRDF
__device__ Color LambertianBRDF::GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const
{
    return _coefficient * _color * Math::InvPi();
}

// get diffuse color
__device__ Color LambertianBRDF::GetDiffuseColor() const
{
    return _color;
}

// get diffuse reflection coefficient
__device__ real_t LambertianBRDF::GetDiffuseCoefficient() const
{
    return _coefficient;
}

// set diffuse color
__device__ void LambertianBRDF::SetDiffuseColor( const Color& color )
{
    _color = color;
}

// set diffuse reflection coefficient
__device__ void LambertianBRDF::SetDiffuseCoefficient( real_t coeff )
{
    _coefficient = coeff;
}

REX_NS_END