#include <rex/Graphics/Lights/AmbientLight.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create ambient light
__device__ AmbientLight::AmbientLight()
    : AmbientLight( Color::White(), 1.0f )
{
}

// create ambient light
__device__ AmbientLight::AmbientLight( const Color& color, real32 ls )
    : Light         ( LightType::Ambient )
    , _radianceScale( ls )
    , _color        ( color )
{
    _castShadows = false;
}

// destroy ambient light
__device__ AmbientLight::~AmbientLight()
{
    _radianceScale = 0.0f;
}

// get color
__device__ const Color& AmbientLight::GetColor() const
{
    return _color;
}

// get light direction
__device__ vec3 AmbientLight::GetLightDirection( ShadePoint& sp ) const
{
    return vec3( 0.0f );
}

// get radiance
__device__ Color AmbientLight::GetRadiance( ShadePoint& sp ) const
{
    return _radianceScale * _color;
}

// get radiance scale
__device__ real32 AmbientLight::GetRadianceScale() const
{
    return _radianceScale;
}

// check if in shadow
__device__ bool AmbientLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
    return false;
}

// set casts shadows
__device__ void AmbientLight::SetCastShadows( bool value )
{
    // do nothing
}

// set color
__device__ void AmbientLight::SetColor( const Color& color )
{
    _color = color;
}

// set color by components
__device__ void AmbientLight::SetColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set radiance scale
__device__ void AmbientLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;
}

REX_NS_END