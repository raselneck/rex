#include <rex/Graphics/Materials/MatteMaterial.hxx>
#include <rex/Graphics/Scene.hxx>

REX_NS_BEGIN

// create material
MatteMaterial::MatteMaterial()
    : MatteMaterial( Color::White(), 0.0f, 0.0f )
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
    SetColor( color );
    SetAmbientCoefficient( ka );
    SetDiffuseCoefficient( kd );
}

// copy material
MatteMaterial::MatteMaterial( const MatteMaterial& other )
{
    _ambient = other._ambient;
    _diffuse = other._diffuse;
}

// destroy material
MatteMaterial::~MatteMaterial()
{
}

// get ka
real32 MatteMaterial::GetAmbientCoefficient() const
{
    return _ambient.GetDiffuseCoefficient();
}

// get color
Color MatteMaterial::GetColor() const
{
    // both ambient and diffuse have the same color
    return _ambient.GetDiffuseColor();
}

// get kd
real32 MatteMaterial::GetDiffuseCoefficient() const
{
    return _diffuse.GetDiffuseCoefficient();
}

// set ka
void MatteMaterial::SetAmbientCoefficient( real32 ka )
{
    _ambient.SetDiffuseCoefficient( ka );
}

// set color
void MatteMaterial::SetColor( const Color& color )
{
    _ambient.SetDiffuseColor( color );
    _diffuse.SetDiffuseColor( color );
}

// set color w/ components
void MatteMaterial::SetColor( real32 r, real32 g, real32 b )
{
    Color color( r, g, b );
    SetColor( color );
}

// set kd
void MatteMaterial::SetDiffuseCoefficient( real32 kd )
{
    _diffuse.SetDiffuseCoefficient( kd );
}

// get shaded color
Color MatteMaterial::Shade( ShadePoint& sp )
{
    // from Suffern, 271

    Vector3 wo      = -sp.Ray.Direction;
    Color   color   = _ambient.GetBHR( sp, wo );
    auto&   lights  = sp.Scene->GetLights();

    for ( auto& light : lights )
    {
        Vector3 wi = light->GetLightDirection( sp );
        real32  angle = static_cast<real32>( Vector3::Dot( sp.Normal, wi ) );

        if ( angle > 0.0 )
        {
            // check if we need to perform shadow calculations
            bool isInShadow = false;
            if ( light->CastsShadows() )
            {
                Ray shadowRay( sp.HitPoint, wi );
                isInShadow = light->IsInShadow( shadowRay, sp );
            }

            // add the shadow-inclusive light information
            if ( !isInShadow )
            {
                Color diffuse = _diffuse.GetBRDF( sp, wo, wi );
                color += diffuse * light->GetRadiance( sp ) * angle;
            }
        }
    }

    return color;
}

REX_NS_END