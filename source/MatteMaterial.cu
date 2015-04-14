#include <rex/Graphics/Materials/MatteMaterial.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create material
__device__ MatteMaterial::MatteMaterial()
    : MatteMaterial( Color::White(), 0.0f, 0.0f, MaterialType::Matte )
{
}

// create material w/ color
__device__ MatteMaterial::MatteMaterial( const Color& color )
    : MatteMaterial( color, 0.0f, 0.0f, MaterialType::Matte )
{
}

// create material w/ color, ambient coefficient, and diffuse coefficient
__device__ MatteMaterial::MatteMaterial( const Color& color, real_t ka, real_t kd )
    : MatteMaterial( color, ka, kd, MaterialType::Matte )
{
}

// create material w/ color, ambient coefficient, diffuse coefficient, and material type
__device__ MatteMaterial::MatteMaterial( const Color& color, real_t ka, real_t kd, MaterialType type )
    : Material( type ),
      _ambient( ka, color ),
      _diffuse( kd, color )
{
}

// destroy material
__device__ MatteMaterial::~MatteMaterial()
{
}

// copy this material
__device__ Material* MatteMaterial::Copy() const
{
    // create the copy of the material
    MatteMaterial* mat = new MatteMaterial( _ambient.GetDiffuseColor(),
                                            _ambient.GetDiffuseCoefficient(),
                                            _diffuse.GetDiffuseCoefficient() );
    return mat;
}

// get ka
__device__ real_t MatteMaterial::GetAmbientCoefficient() const
{
    return _ambient.GetDiffuseCoefficient();
}

// get color
__device__ Color MatteMaterial::GetColor() const
{
    // both ambient and diffuse have the same color
    return _ambient.GetDiffuseColor();
}

// get kd
__device__ real_t MatteMaterial::GetDiffuseCoefficient() const
{
    return _diffuse.GetDiffuseCoefficient();
}

// set ka
__device__ void MatteMaterial::SetAmbientCoefficient( real_t ka )
{
    _ambient.SetDiffuseCoefficient( ka );
}

// set color
__device__ void MatteMaterial::SetColor( const Color& color )
{
    _ambient.SetDiffuseColor( color );
    _diffuse.SetDiffuseColor( color );
}

// set color w/ components
__device__ void MatteMaterial::SetColor( real_t r, real_t g, real_t b )
{
    Color color = Color( r, g, b );
    SetColor( color );
}

// set kd
__device__ void MatteMaterial::SetDiffuseCoefficient( real_t kd )
{
    _diffuse.SetDiffuseCoefficient( kd );
}

// get shaded color
__device__ Color MatteMaterial::Shade( ShadePoint& sp, const DeviceList<Light*>* lights, const Octree* octree ) const
{
    // from Suffern, 271
    Vector3 wo    = -sp.Ray.Direction;
    Color   color = _ambient.GetBHR( sp, wo );

    // go through all of the lights in the scene
    for ( uint_t i = 0; i < lights->GetSize(); ++i )
    {
        const Light*  light = lights->operator[]( i );
        Vector3       wi    = light->GetLightDirection( sp );
        real_t        angle = static_cast<real_t>( Vector3::Dot( sp.Normal, wi ) );

        if ( angle > 0.0 )
        {
            // check if we need to perform shadow calculations
            bool isInShadow = false;
            if ( light->CastsShadows() )
            {
                Ray shadowRay = Ray( sp.HitPoint, wi );
                isInShadow = light->IsInShadow( shadowRay, octree, sp );
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