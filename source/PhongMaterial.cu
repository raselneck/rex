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

// get area light shaded color
__device__ Color PhongMaterial::AreaLightShade( ShadePoint& sp ) const
{
    // adapted from Suffern, 332
    vec3  wo    = -sp.Ray.Direction;
    Color color = _ambient.GetBHR( sp, wo ) * sp.AmbientLight->GetRadiance( sp );

    for ( uint32 i = 0; i < sp.LightCount; ++i )
    {
        const Light* light  = sp.Lights[ i ];
        vec3         wi     = light->GetLightDirection( sp );
        real32       angle  = glm::dot( sp.Normal, wi );
        
        if ( angle > 0.0f )
        {
            // calculate shadow information
            Ray   shadowRay   = Ray( sp.HitPoint, wi );
            bool  isInShadow  = light->CastsShadows() && light->IsInShadow( shadowRay, sp );
            Color diffuse     = _diffuse.GetBRDF( sp, wo, wi );
            Color specular    = _specular.GetBRDF( sp, wo, wi );
            Color shadowColor = ( diffuse + specular ) * light->GetRadiance( sp ) * angle
                              * light->GetGeometricFactor( sp ) * light->GetGeometricArea( sp );

            // calculate the color with a branchless conditional
            color += Color::Lerp( shadowColor,
                                  Color::Black(),
                                  static_cast<real32>( isInShadow ) );
        }
    }

    return color;
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

// get shaded color
__device__ Color PhongMaterial::Shade( ShadePoint& sp ) const
{
    // adapted from Suffern, 285
    vec3  wo    = -sp.Ray.Direction;
    Color color = _ambient.GetBHR( sp, wo ) * sp.AmbientLight->GetRadiance( sp );

    for ( uint32 i = 0; i < sp.LightCount; ++i )
    {
        const Light* light  = sp.Lights[ i ];
        vec3         wi     = light->GetLightDirection( sp );
        real32       angle  = glm::dot( sp.Normal, wi );
        
        if ( angle > 0.0f )
        {
            // calculate shadow information
            Ray   shadowRay   = Ray( sp.HitPoint, wi );
            bool  isInShadow  = light->CastsShadows() && light->IsInShadow( shadowRay, sp );
            Color diffuse     = _diffuse.GetBRDF( sp, wo, wi );
            Color specular    = _specular.GetBRDF( sp, wo, wi );
            Color shadowColor = ( diffuse + specular ) * light->GetRadiance( sp ) * angle;

            // calculate the color with a branchless conditional
            color += Color::Lerp( shadowColor,
                                  Color::Black(),
                                  static_cast<real32>( isInShadow ) );
        }
    }

    return color;
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

REX_NS_END