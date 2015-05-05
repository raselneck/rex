#include <rex/Graphics/Lights/PointLight.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

// TODO : Implement point light attenuation

REX_NS_BEGIN

// create point light
__device__ PointLight::PointLight()
    : PointLight( vec3( 0.0, 0.0, 0.0 ) )
{
}

// create point light w/ position components
__device__ PointLight::PointLight( real32 x, real32 y, real32 z )
    : PointLight( vec3( x, y, z ) )
{
}

// create point light w/ position
__device__ PointLight::PointLight( const vec3& position )
    : Light( LightType::Point ),
      _position( position ),
      _color( Color::White() ),
      _radianceScale( 1.0f )
{
    _castShadows = true;
}

// destroy point light
__device__ PointLight::~PointLight()
{
    _radianceScale = 0.0f;
}

// get color
__device__ const Color& PointLight::GetColor() const
{
    return _color;
}

// get light direction
__device__ vec3 PointLight::GetLightDirection( ShadePoint& sp ) const
{
    return glm::normalize( _position - sp.HitPoint );
}

// get position
__device__ const vec3& PointLight::GetPosition() const
{
    return _position;
}

// get radiance
__device__ Color PointLight::GetRadiance( ShadePoint& sp ) const
{
    return _radianceScale * _color;
}

// get radiance scale
__device__ real32 PointLight::GetRadianceScale() const
{
    return _radianceScale;
}

// check if in shadow
__device__ bool PointLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
    // based on Suffern, 300

    real32 t = 0.0;
    real32 d = glm::distance( _position, ray.Origin );

    return sp.Octree->QueryShadowRay( ray, t ) && ( t < d );
}

// set color
__device__ void PointLight::SetColor( const Color& color )
{
    _color = color;
}

// set color components
__device__ void PointLight::SetColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.B = g;
    _color.B = b;
}

// set position
__device__ void PointLight::SetPosition( const vec3& position )
{
    _position = position;
}

// set position
__device__ void PointLight::SetPosition( real32 x, real32 y, real32 z )
{
    _position.x = x;
    _position.y = y;
    _position.z = z;
}

// set radiance scale
__device__ void PointLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;
}

REX_NS_END