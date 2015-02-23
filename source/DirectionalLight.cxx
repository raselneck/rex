#include <rex/Lights/DirectionalLight.hxx>
#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

// create light
DirectionalLight::DirectionalLight()
    : DirectionalLight( Vector3() )
{
}

// create light w/ direction
DirectionalLight::DirectionalLight( const Vector3& direction )
    : _direction( Vector3::Normalize( direction ) ), _radianceScale( 1.0f ), _color( Color::White )
{
}

// create light w/ direction components
DirectionalLight::DirectionalLight( real64 x, real64 y, real64 z )
    : DirectionalLight( Vector3( x, y, z ) )
{
}

// destroy light
DirectionalLight::~DirectionalLight()
{
    _radianceScale = 0.0f;
}

// get color
const Color& DirectionalLight::GetColor() const
{
    return _color;
}

// get direction
const Vector3& DirectionalLight::GetDirection() const
{
    return _direction;
}

// get radiance scale
real32 DirectionalLight::GetRadianceScale() const
{
    return _radianceScale;
}

// get direction of incoming light
Vector3 DirectionalLight::GetLightDirection( ShadePoint& sp )
{
    return _direction;
}

// get radiance
Color DirectionalLight::GetRadiance( ShadePoint& sp )
{
    return _radianceScale * _color;
}

// set color
void DirectionalLight::SetColor( const Color& color )
{
    _color = color;
}

// set color w/ components
void DirectionalLight::SetColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set direction
void DirectionalLight::SetDirection( const Vector3& direction )
{
    _direction = Vector3::Normalize( direction );
}

// set direction w/ components
void DirectionalLight::SetDirection( real64 x, real64 y, real64 z )
{
    SetDirection( Vector3( x, y, z ) );
}

// set radiance scale
void DirectionalLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;
}

REX_NS_END