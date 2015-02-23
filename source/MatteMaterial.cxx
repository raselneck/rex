#include <rex/Materials/MatteMaterial.hxx>
#include <rex/Scene/Scene.hxx>

REX_NS_BEGIN

// create material
MatteMaterial::MatteMaterial()
    : MatteMaterial( Color::White )
{
}

// create material w/ color
MatteMaterial::MatteMaterial( const Color& color )
    : MatteMaterial( color, 0.0f, 0.0f )
{
}

// create material w/ color, ambient coefficient, diffuse coefficient
MatteMaterial::MatteMaterial( const Color& color, real32 ka, real32 kd )
{
    _ambient.reset( new LambertianBRDF() );
    _diffuse.reset( new LambertianBRDF() );

    SetColor( color );
    SetAmbientCoefficient( ka );
    SetDiffuseCoefficient( kd );
}

// destroy material
MatteMaterial::~MatteMaterial()
{
}

// get ka
real32 MatteMaterial::GetAmbientCoefficient() const
{
    return _ambient->GetDiffuseCoefficient();
}

// get color
Color MatteMaterial::GetColor() const
{
    // both ambient and diffuse have the same color
    return _ambient->GetDiffuseColor();
}

// get kd
real32 MatteMaterial::GetDiffuseCoefficient() const
{
    return _diffuse->GetDiffuseCoefficient();
}

// set ka
void MatteMaterial::SetAmbientCoefficient( real32 ka )
{
    _ambient->SetDiffuseCoefficient( ka );
}

// set color
void MatteMaterial::SetColor( const Color& color )
{
    _ambient->SetDiffuseColor( color );
    _diffuse->SetDiffuseColor( color );
}

// set kd
void MatteMaterial::SetDiffuseCoefficient( real32 kd )
{
    _diffuse->SetDiffuseCoefficient( kd );
}

// set sampler
void MatteMaterial::SetSampler( Handle<Sampler>& sampler )
{
    _ambient->SetSampler( sampler );
    _diffuse->SetSampler( sampler );
}

// get shaded color
Color MatteMaterial::Shade( ShadePoint& sp )
{
    // from Suffern, 271

    Vector3 wo         = -sp.Ray.Direction;
    Color   color      = _ambient->GetBHR( sp, wo );
    auto&   lights     = sp.Scene->GetLights();
    uint32  lightCount = sp.Scene->GetLightCount();

    for ( uint32 i = 0; i < lightCount; ++i )
    {
        Vector3 wi = lights[ i ]->GetLightDirection( sp );
        real32  angle = static_cast<real32>( Vector3::Dot( sp.Normal, wi ) );

        if ( angle > 0.0 )
        {
            Color diffuse = _diffuse->GetBRDF( sp, wi, wo );
            color += diffuse * lights[ i ]->GetRadiance( sp ) * angle;
        }
    }

    return color;
}

REX_NS_END