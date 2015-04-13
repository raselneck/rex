#include <rex/Graphics/Lights/PointLight.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

// TODO : Implement point light attenuation

REX_NS_BEGIN

// create point light
__device__ PointLight::PointLight()
    : PointLight( Vector3( 0.0, 0.0, 0.0 ) )
{
}

// create point light w/ position components
__device__ PointLight::PointLight( real_t x, real_t y, real_t z )
    : PointLight( Vector3( x, y, z ) )
{
}

// create point light w/ position
__device__ PointLight::PointLight( const Vector3& position )
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
__device__ Vector3 PointLight::GetLightDirection( ShadePoint& sp ) const
{
    return Vector3::Normalize( _position - sp.HitPoint );
}

// get position
__device__ const Vector3& PointLight::GetPosition() const
{
    return _position;
}

// get radiance
__device__ Color PointLight::GetRadiance( ShadePoint& sp ) const
{
    return _radianceScale * _color;
}

// get radiance scale
__device__ real_t PointLight::GetRadianceScale() const
{
    return _radianceScale;
}

// check if in shadow
__device__ bool PointLight::IsInShadow( const Ray& ray, const Octree* octree, const ShadePoint& sp ) const
{
    // based on Suffern, 300

    real_t t = 0.0;
    real_t d = Vector3::Distance( _position, ray.Origin );

    if ( octree->QueryShadowRay( ray, t ) && ( t < d ) )
    {
        return true;
    }

    return false;
}

// set color
__device__ void PointLight::SetColor( const Color& color )
{
    _color = color;
}

// set color components
__device__ void PointLight::SetColor( real_t r, real_t g, real_t b )
{
    _color.R = r;
    _color.B = g;
    _color.B = b;
}

// set position
__device__ void PointLight::SetPosition( const Vector3& position )
{
    _position = position;
}

// set position
__device__ void PointLight::SetPosition( real_t x, real_t y, real_t z )
{
    _position.X = x;
    _position.Y = y;
    _position.Z = z;
}

// set radiance scale
__device__ void PointLight::SetRadianceScale( real_t ls )
{
    _radianceScale = ls;
}

REX_NS_END