#include <rex/Graphics/Materials/PhongMaterial.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create material
__device__ PhongMaterial::PhongMaterial()
    : PhongMaterial( Color::White(), 0.0f, 0.0f, 0.0f, 0.0f )
{
}

// create material w/ color
__device__ PhongMaterial::PhongMaterial( const Color& color )
    : PhongMaterial( color, 0.0f, 0.0f, 0.0f, 0.0f )
{
}

// create material w/ color, ambient coefficient, diffuse coefficient, specular coefficient, specular power
__device__ PhongMaterial::PhongMaterial( const Color& color, real32 ka, real32 kd, real32 ks, real32 pow )
    : MatteMaterial( color, ka, kd, MaterialType::Phong ),
      _specular( ks, color, pow )
{
}

// destroy material
__device__ PhongMaterial::~PhongMaterial()
{
}

// copy this material
__device__ Material* PhongMaterial::Copy() const
{
    // create the copy of the material
    PhongMaterial* mat = new PhongMaterial( _ambient.GetDiffuseColor(),
                                            _ambient.GetDiffuseCoefficient(),
                                            _diffuse.GetDiffuseCoefficient(),
                                            _specular.GetSpecularCoefficient(),
                                            _specular.GetSpecularPower() );
    return mat;
}

// get specular coefficient
__device__ real32 PhongMaterial::GetSpecularCoefficient() const
{
    return _specular.GetSpecularCoefficient();
}

// get specular power
__device__ real32 PhongMaterial::GetSpecularPower() const
{
    return _specular.GetSpecularPower();
}

// set ambient coefficient
__device__ void PhongMaterial::SetAmbientCoefficient( real32 ka )
{
    _ambient.SetDiffuseCoefficient( ka );
}

// set color
__device__ void PhongMaterial::SetColor( const Color& color )
{
    _ambient.SetDiffuseColor( color );
    _diffuse.SetDiffuseColor( color );
    _specular.SetSpecularColor( color );
}

// set color w/ components
__device__ void PhongMaterial::SetColor( real32 r, real32 g, real32 b )
{
    Color color = Color( r, g, b );
    SetColor( color );
}

// set diffuse coefficient
__device__ void PhongMaterial::SetDiffuseCoefficient( real32 kd )
{
    _diffuse.SetDiffuseCoefficient( kd );
}

// set specular coefficient
__device__ void PhongMaterial::SetSpecularCoefficient( real32 ks )
{
    _specular.SetSpecularCoefficient( ks );
}

// set specular power
__device__ void PhongMaterial::SetSpecularPower( real32 pow )
{
    _specular.SetSpecularPower( pow );
}

// get shaded color
__device__ Color PhongMaterial::Shade( ShadePoint& sp, const DeviceList<Light*>* lights, const Octree* octree ) const
{
    // from Suffern, 285
    vec3 wo    = -sp.Ray.Direction;
    Color   color = _ambient.GetBHR( sp, wo );

    for ( uint32 i = 0; i < lights->GetSize(); ++i )
    {
        const Light* light = lights->Get( i );
        vec3         wi    = light->GetLightDirection( sp );
        real32       angle = glm::dot( sp.Normal, wi );

        if ( angle > 0.0 )
        {
            // check if we need to perform shadow calculations
            bool isInShadow = false;
            if ( light->CastsShadows() )
            {
                Ray shadowRay( sp.HitPoint, wi );
                isInShadow = light->IsInShadow( shadowRay, octree, sp );
            }

            // add the shadow-inclusive light information
            if ( !isInShadow )
            {
                Color diffuse   = _diffuse.GetBRDF( sp, wo, wi );
                Color specular  = _specular.GetBRDF( sp, wo, wi );
                color          += ( diffuse + specular ) * light->GetRadiance( sp ) * angle;
            }
        }
    }

    return color;
}

REX_NS_END