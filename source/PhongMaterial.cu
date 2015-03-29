#include <rex/Graphics/Materials/PhongMaterial.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create material
PhongMaterial::PhongMaterial()
    : PhongMaterial( Color::White(), 0.0f, 0.0f, 0.0f, 0.0f )
{
}

// create material w/ color
PhongMaterial::PhongMaterial( const Color& color )
    : PhongMaterial( color, 0.0f, 0.0f, 0.0f, 0.0f )
{
}

// create material w/ color, ambient coefficient, diffuse coefficient, specular coefficient, specular power
PhongMaterial::PhongMaterial( const Color& color, real32 ka, real32 kd, real32 ks, real32 pow )
    : MatteMaterial( color, ka, kd, false ),
      _specular( ks, color, pow )
{
    // allocate us on the device
    _dThis = GC::DeviceAlloc<PhongMaterial>( *this );
}

// destroy material
PhongMaterial::~PhongMaterial()
{
}

// get specular coefficient
real32 PhongMaterial::GetSpecularCoefficient() const
{
    return _specular.GetSpecularCoefficient();
}

// get specular power
real32 PhongMaterial::GetSpecularPower() const
{
    return _specular.GetSpecularPower();
}

// get material type
MaterialType PhongMaterial::GetType() const
{
    return MaterialType::Phong;
}

// set ambient coefficient
void PhongMaterial::SetAmbientCoefficient( real32 ka )
{
    _ambient.SetDiffuseCoefficient( ka );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PhongMaterial ), cudaMemcpyHostToDevice );
}

// set color
void PhongMaterial::SetColor( const Color& color )
{
    _ambient.SetDiffuseColor( color );
    _diffuse.SetDiffuseColor( color );
    _specular.SetSpecularColor( color );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PhongMaterial ), cudaMemcpyHostToDevice );
}

// set color w/ components
void PhongMaterial::SetColor( real32 r, real32 g, real32 b )
{
    SetColor( Color( r, g, b ) );
}

// set diffuse coefficient
void PhongMaterial::SetDiffuseCoefficient( real32 kd )
{
    _diffuse.SetDiffuseCoefficient( kd );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PhongMaterial ), cudaMemcpyHostToDevice );
}

// set specular coefficient
void PhongMaterial::SetSpecularCoefficient( real32 ks )
{
    _specular.SetSpecularCoefficient( ks );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PhongMaterial ), cudaMemcpyHostToDevice );
}

// set specular power
void PhongMaterial::SetSpecularPower( real32 pow )
{
    _specular.SetSpecularPower( pow );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PhongMaterial ), cudaMemcpyHostToDevice );
}

// get shaded color
Color PhongMaterial::Shade( ShadePoint& sp )
{
    // from Suffern, 285

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
                Color diffuse   = _diffuse.GetBRDF( sp, wo, wi );
                Color specular  = _specular.GetBRDF( sp, wo, wi );
                color += ( diffuse + specular ) * light->GetRadiance( sp ) * angle;
            }
        }
    }

    return color;
}

REX_NS_END