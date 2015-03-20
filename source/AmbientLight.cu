#include <rex/Graphics/Lights/AmbientLight.hxx>
#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// create ambient light
AmbientLight::AmbientLight()
: _radianceScale( 1.0f ), _color( Color::White() )
{
    _castShadows = false;
}

// destroy ambient light
AmbientLight::~AmbientLight()
{
    _radianceScale = 0.0f;
}

// get color
const Color& AmbientLight::GetColor() const
{
    return _color;
}

// get radiance scale
real32 AmbientLight::GetRadianceScale() const
{
    return _radianceScale;
}

// get light direction
Vector3 AmbientLight::GetLightDirection( ShadePoint& sp )
{
    return Vector3( 0.0 );
}

// get radiance
Color AmbientLight::GetRadiance( ShadePoint& sp )
{
    return _radianceScale * _color;
}

// check if in shadow
bool AmbientLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
    return false;
}

// set casts shadows
void AmbientLight::SetCastShadows( bool value )
{
    // do nothing
}

// set color
void AmbientLight::SetColor( const Color& color )
{
    _color = color;
}

// set color by components
void AmbientLight::SetColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set radiance scale
void AmbientLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;
}

REX_NS_END