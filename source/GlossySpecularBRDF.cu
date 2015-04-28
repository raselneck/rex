#include <rex/Graphics/BRDFs/GlossySpecularBRDF.hxx>

REX_NS_BEGIN

// create g-s BRDF
__device__ GlossySpecularBRDF::GlossySpecularBRDF()
    : GlossySpecularBRDF( 0.0f, Color::Black(), 0.0f )
{
}

// create g-s BRDF w/ coefficient, color, power
__device__ GlossySpecularBRDF::GlossySpecularBRDF( real32 ks, const Color& color, real32 pow )
    : _coefficient( ks )
    , _color      ( color )
    , _power      ( pow )
{
}

// destroy gs- BRDF
__device__ GlossySpecularBRDF::~GlossySpecularBRDF()
{
    _coefficient = 0.0f;
    _power = 0.0f;
}

// get bi-hemispherical reflectance (rho)
__device__ Color GlossySpecularBRDF::GetBHR( const ShadePoint& sp, const vec3& wo ) const
{
    return Color::Magenta();
}

// get BRDF (f)
__device__ Color GlossySpecularBRDF::GetBRDF( const ShadePoint& sp, const vec3& wo, const vec3& wi ) const
{
    // from Suffern, 284

    Color  color;
    vec3   reflected = -wi + 2.0f * sp.Normal * dot( sp.Normal, wi );

    // BRANCHLESS CONDITIONAL, WOOOO
    real32 d = dot( reflected, wo );
    /* if ( d > 0.0f )
    {
        color = _coefficient * _color * pow( d, _power );
    } */
    color = Color::Lerp( color,
                         _coefficient * _color * pow( d, _power ),
                         d > 0.0f );
    

    return color;
}

// get specular coefficient
__device__ real32 GlossySpecularBRDF::GetSpecularCoefficient() const
{
    return _coefficient;
}

// get color
__device__ const Color& GlossySpecularBRDF::GetSpecularColor() const
{
    return _color;
}

// get power
__device__ real32 GlossySpecularBRDF::GetSpecularPower() const
{
    return _power;
}

// set ks
__device__ void GlossySpecularBRDF::SetSpecularCoefficient( real32 ks )
{
    _coefficient = ks;
}

// set color
__device__ void GlossySpecularBRDF::SetSpecularColor( const Color& color )
{
    _color = color;
}

// set color w/ components
__device__ void GlossySpecularBRDF::SetSpecularColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set power
__device__ void GlossySpecularBRDF::SetSpecularPower( real32 pow )
{
    _power = pow;
}

REX_NS_END