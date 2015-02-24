#include <rex/Lights/PointLight.hxx>
#include <rex/Scene/Scene.hxx>
#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

// create point light
PointLight::PointLight()
    : PointLight( 0.0, 0.0, 0.0 )
{
}

// create point light w/ position
PointLight::PointLight( const Vector3& position )
    : _position( position ), _color( Color::White ), _radianceScale( 1.0f )
{
    _castShadows = true;
}

// create point light w/ position components
PointLight::PointLight( real64 x, real64 y, real64 z )
    : _position( x, y, z ), _color( Color::White ), _radianceScale( 1.0f )
{
}

// destroy point light
PointLight::~PointLight()
{
    _radianceScale = 0.0f;
}

// get color
const Color& PointLight::GetColor() const
{
    return _color;
}

// get position
const Vector3& PointLight::GetPosition() const
{
    return _position;
}

// get radiance scale
real32 PointLight::GetRadianceScale() const
{
    return _radianceScale;
}

// get light direction
Vector3 PointLight::GetLightDirection( ShadePoint& sp )
{
    return Vector3::Normalize( _position - sp.HitPoint );
}

// get radiance
Color PointLight::GetRadiance( ShadePoint& sp )
{
    return _radianceScale * _color;
}

// check if in shadow
bool PointLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
    // from Suffern, 300

    real64 t = 0.0;
    real64 d = Vector3::Distance( _position, ray.Origin );

    for ( auto& obj : sp.Scene->GetObjects() )
    {
        if ( obj->ShadowHit( ray, t ) && ( t < d ) )
        {
            return true;
        }
    }

    return false;
}

// set color
void PointLight::SetColor( const Color& color )
{
    _color = color;
}

// set color components
void PointLight::SetColor( real32 r, real32 g, real32 b )
{
    _color.R = r;
    _color.G = g;
    _color.B = b;
}

// set position
void PointLight::SetPosition( const Vector3& position )
{
    _position = position;
}

// set position
void PointLight::SetPosition( real64 x, real64 y, real64 z )
{
    _position.X = x;
    _position.Y = y;
    _position.Z = z;
}

// set radiance scale
void PointLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;
}

REX_NS_END