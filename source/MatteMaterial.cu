#include <rex/Graphics/Materials/MatteMaterial.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create material
MatteMaterial::MatteMaterial()
    : MatteMaterial( Color::White(), 0.0f, 0.0f, true )
{
}

// create material w/ color
MatteMaterial::MatteMaterial( const Color& color )
    : MatteMaterial( color, 0.0f, 0.0f, true )
{
}

// create material w/ color, ambient coefficient, diffuse coefficient
MatteMaterial::MatteMaterial( const Color& color, real32 ka, real32 kd )
    : MatteMaterial( color, ka, kd, true )
{
}

// create material w/ color, ambient coefficient, diffuse coefficient
MatteMaterial::MatteMaterial( const Color& color, real32 ka, real32 kd, bool allocOnDevice )
    : _dThis( nullptr )
{
    // set ambient info
    _ambient.SetDiffuseColor( color );
    _ambient.SetDiffuseCoefficient( ka );

    // set diffuse info
    _diffuse.SetDiffuseColor( color );
    _diffuse.SetDiffuseCoefficient( kd );

    // check if we need to allocate us on the device
    if ( allocOnDevice )
    {
        _dThis = GC::DeviceAlloc<MatteMaterial>( *this );
    }
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

// get material on device
const Material* MatteMaterial::GetOnDevice() const
{
    return static_cast<const Material*>( _dThis );
}

// get material type
MaterialType MatteMaterial::GetType() const
{
    return MaterialType::Matte;
}

// set ka
void MatteMaterial::SetAmbientCoefficient( real32 ka )
{
    _ambient.SetDiffuseCoefficient( ka );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( MatteMaterial ), cudaMemcpyHostToDevice );
}

// set color
void MatteMaterial::SetColor( const Color& color )
{
    _ambient.SetDiffuseColor( color );
    _diffuse.SetDiffuseColor( color );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( MatteMaterial ), cudaMemcpyHostToDevice );
}

// set color w/ components
void MatteMaterial::SetColor( real32 r, real32 g, real32 b )
{
    SetColor( Color( r, g, b ) );
}

// set kd
void MatteMaterial::SetDiffuseCoefficient( real32 kd )
{
    _diffuse.SetDiffuseCoefficient( kd );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( MatteMaterial ), cudaMemcpyHostToDevice );
}

// get shaded color
__device__ Color MatteMaterial::Shade( ShadePoint& sp, const Light** lights, uint32 lightCount )
{
    // from Suffern, 271
    Vector3 wo    = -sp.Ray.Direction;
    Color   color = _ambient.GetBHR( sp, wo );

    // go through all of the lights in the scene
    for ( uint32 i = 0; i < lightCount; ++i )
    {
        Light*  light = const_cast<Light*>( lights[ i ] ); // TODO : BAD
        Vector3 wi    = light->GetLightDirection( sp );
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