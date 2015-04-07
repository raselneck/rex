#include <rex/Graphics/BRDFs/GlossySpecularBRDF.hxx>

REX_NS_BEGIN

// create g-s BRDF
GlossySpecularBRDF::GlossySpecularBRDF()
: _ks( 0.0f ), _color( Color::White() ), _pow( 0.0f )
{
}

// create g-s BRDF w/ coefficient, color, power
GlossySpecularBRDF::GlossySpecularBRDF( real32 ks, const Color& color, real32 pow )
    : _ks( ks ), _color( color ), _pow( pow )
{
}

// destroy gs- BRDF
GlossySpecularBRDF::~GlossySpecularBRDF()
{
    _ks = 0.0f;
    _pow = 0.0f;
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
    real32  angle = static_cast<real32>( Vector3::Dot( sp.Normal, wi ) );
    Vector3 reflected = -wi + 2.0 * sp.Normal * angle;

    angle = static_cast<real32>( Vector3::Dot( reflected, wo ) );
    if ( angle > 0.0 )
    {
        color = _ks * _color * pow( angle, _pow );
    }

    return color;
}

// get ks
real32 GlossySpecularBRDF::GetSpecularCoefficient() const
{
    return _ks;
}

// get color
const Color& GlossySpecularBRDF::GetSpecularColor() const
{
    return _color;
}

// get power
real32 GlossySpecularBRDF::GetSpecularPower() const
{
    return _pow;
}

// set ks
void GlossySpecularBRDF::SetSpecularCoefficient( real32 ks )
{
    _ks = ks;
}

// set color
void GlossySpecularBRDF::SetSpecularColor( const Color& color )
{
    _color = color;
}

// set color w/ components
void GlossySpecularBRDF::SetSpecularColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set power
void GlossySpecularBRDF::SetSpecularPower( real32 pow )
{
    _pow = pow;
}

REX_NS_END