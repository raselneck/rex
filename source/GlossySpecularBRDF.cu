#include <rex/Graphics/BRDFs/GlossySpecularBRDF.hxx>

REX_NS_BEGIN

// create g-s BRDF
__device__ GlossySpecularBRDF::GlossySpecularBRDF()
    : GlossySpecularBRDF( 0.0f, Color::Black(), 0.0f )
{
}

// create g-s BRDF w/ coefficient, color, power
__device__ GlossySpecularBRDF::GlossySpecularBRDF( real_t ks, const Color& color, real_t pow )
    : _coefficient( ks ),
      _color      ( color ),
      _power      ( pow )
{
}

// destroy gs- BRDF
__device__ GlossySpecularBRDF::~GlossySpecularBRDF()
{
    _coefficient = 0.0f;
    _power = 0.0f;
}

// get bi-hemispherical reflectance (rho)
__device__ Color GlossySpecularBRDF::GetBHR( const ShadePoint& sp, const Vector3& wo ) const
{
    return Color::Magenta();
}

// get BRDF (f)
__device__ Color GlossySpecularBRDF::GetBRDF( const ShadePoint& sp, const Vector3& wo, const Vector3& wi ) const
{
    // from Suffern, 284

    Color   color;
    real_t  angle     = Vector3::Dot( sp.Normal, wi );
    Vector3 reflected = -wi + 2.0 * sp.Normal * angle;

    angle = Vector3::Dot( reflected, wo );
    if ( angle > 0.0 )
    {
        color = _coefficient * _color * pow( angle, _power );
    }

    return color;
}

// get ks
__device__ real_t GlossySpecularBRDF::GetSpecularCoefficient() const
{
    return _coefficient;
}

// get color
__device__ const Color& GlossySpecularBRDF::GetSpecularColor() const
{
    return _color;
}

// get power
__device__ real_t GlossySpecularBRDF::GetSpecularPower() const
{
    return _power;
}

// set ks
__device__ void GlossySpecularBRDF::SetSpecularCoefficient( real_t ks )
{
    _coefficient = ks;
}

// set color
__device__ void GlossySpecularBRDF::SetSpecularColor( const Color& color )
{
    _color = color;
}

// set color w/ components
__device__ void GlossySpecularBRDF::SetSpecularColor( real_t r, real_t g, real_t b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set power
__device__ void GlossySpecularBRDF::SetSpecularPower( real_t pow )
{
    _power = pow;
}

REX_NS_END