#include <rex/Graphics/Lights/DirectionalLight.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create light
__device__ DirectionalLight::DirectionalLight()
    : DirectionalLight( vec3( 0.0f, -1.0f, 0.0f ) )
{
}

// create light w/ direction components
__device__ DirectionalLight::DirectionalLight( real32 x, real32 y, real32 z )
    : DirectionalLight( vec3( x, y, z ) )
{
}

// create light w/ direction
__device__ DirectionalLight::DirectionalLight( const vec3& direction )
    : Light         ( LightType::Directional      )
    , _direction    ( glm::normalize( direction ) )
    , _color        ( Color::White()              )
    , _radianceScale( 1.0f                        )
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
__device__ const vec3& DirectionalLight::GetDirection() const
{
    return _direction;
}

// get direction of incoming light
__device__ vec3 DirectionalLight::GetLightDirection( ShadePoint& sp ) const
{
    return _direction;
}

// get radiance
__device__ Color DirectionalLight::GetRadiance( ShadePoint& sp ) const
{
    return _radianceScale * _color;
}

// get radiance scale
__device__ real32 DirectionalLight::GetRadianceScale() const
{
    return _radianceScale;
}

// check if in shadow
__device__ bool DirectionalLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
    // I'm guessing at this implementation, as Suffern does not provide one.
    // it seems to work, so if the glove fits...

    const vec3 myRayOffset = _direction * 0.001f;
    Ray    myRay = Ray( ray.Origin + myRayOffset, _direction );
    real32 d     = 0.0;
    return sp.Octree->QueryShadowRay( myRay, d );
}

// set color
__device__ void DirectionalLight::SetColor( const Color& color )
{
    _color = color;
}

// set color w/ components
__device__ void DirectionalLight::SetColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set direction
__device__ void DirectionalLight::SetDirection( const vec3& direction )
{
    _direction = glm::normalize( direction );
}

// set direction w/ components
__device__ void DirectionalLight::SetDirection( real32 x, real32 y, real32 z )
{
    vec3 dir = vec3( x, y, z );
    SetDirection( dir );
}

// set radiance scale
__device__ void DirectionalLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;
}

REX_NS_END