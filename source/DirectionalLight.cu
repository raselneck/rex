#include <rex/Graphics/Lights/DirectionalLight.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create light
__device__ DirectionalLight::DirectionalLight()
    : DirectionalLight( Vector3() )
{
}

// create light w/ direction components
__device__ DirectionalLight::DirectionalLight( real_t x, real_t y, real_t z )
    : DirectionalLight( Vector3( x, y, z ) )
{
}

// create light w/ direction
__device__ DirectionalLight::DirectionalLight( const Vector3& direction )
    : Light( LightType::Directional ),
      _direction( Vector3::Normalize( direction ) ),
      _color( Color::White() ),
      _radianceScale( 1.0f )
{
    _castShadows = true;
}

// destroy light
__device__ DirectionalLight::~DirectionalLight()
{
    _radianceScale = 0.0f;
}

// get color
__device__ const Color& DirectionalLight::GetColor() const
{
    return _color;
}

// get direction
__device__ const Vector3& DirectionalLight::GetDirection() const
{
    return _direction;
}

// get direction of incoming light
__device__ Vector3 DirectionalLight::GetLightDirection( ShadePoint& sp ) const
{
    return _direction;
}

// get radiance
__device__ Color DirectionalLight::GetRadiance( ShadePoint& sp ) const
{
    return _radianceScale * _color;
}

// get radiance scale
__device__ real_t DirectionalLight::GetRadianceScale() const
{
    return _radianceScale;
}

// check if in shadow
__device__ bool DirectionalLight::IsInShadow( const Ray& ray, const Octree* octree, const ShadePoint& sp ) const
{
    // I'm guessing at this implementation, as Suffern does not provide one.
    // it seems to work, so if the glove fits...

    Ray    myRay = Ray( ray.Origin + _direction * 10, _direction );
    real_t d     = 0.0;
    return octree->QueryShadowRay( myRay, d );
}

// set color
__device__ void DirectionalLight::SetColor( const Color& color )
{
    _color = color;
}

// set color w/ components
__device__ void DirectionalLight::SetColor( real_t r, real_t g, real_t b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set direction
__device__ void DirectionalLight::SetDirection( const Vector3& direction )
{
    _direction = Vector3::Normalize( direction );
}

// set direction w/ components
__device__ void DirectionalLight::SetDirection( real_t x, real_t y, real_t z )
{
    Vector3 dir = Vector3( x, y, z );
    SetDirection( dir );
}

// set radiance scale
__device__ void DirectionalLight::SetRadianceScale( real_t ls )
{
    _radianceScale = ls;
}

REX_NS_END