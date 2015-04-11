#include <rex/Graphics/Lights/PointLight.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

// TODO : Implement point light attenuation
// TODO : Override SetCastsShadows to update on device

REX_NS_BEGIN

// create point light
PointLight::PointLight()
    : PointLight( Vector3( 0.0, 0.0, 0.0 ) )
{
}

// create point light w/ position components
PointLight::PointLight( real64 x, real64 y, real64 z )
    : PointLight( Vector3( x, y, z ) )
{
}

// create point light w/ position
PointLight::PointLight( const Vector3& position )
    : _position( position ),
      _color( Color::White() ),
      _radianceScale( 1.0f )
{
    _castShadows = true;

    // create us on the device
    _dThis = GC::DeviceAlloc<PointLight>( *this );
}

// destroy point light
PointLight::~PointLight()
{
    _radianceScale = 0.0f;
    _dThis         = nullptr;
}

// get color
const Color& PointLight::GetColor() const
{
    return _color;
}

// get light direction
__device__ Vector3 PointLight::GetLightDirection( ShadePoint& sp ) const
{
    return Vector3::Normalize( _position - sp.HitPoint );
}

// get light on the device
const Light* PointLight::GetOnDevice() const
{
    return static_cast<Light*>( _dThis );
}

// get position
const Vector3& PointLight::GetPosition() const
{
    return _position;
}

// get radiance
__device__ Color PointLight::GetRadiance( ShadePoint& sp ) const
{
    return _radianceScale * _color;
}

// get radiance scale
real32 PointLight::GetRadianceScale() const
{
    return _radianceScale;
}

// get type
LightType PointLight::GetType() const
{
    return LightType::PointLight;
}

// check if in shadow
__device__ bool PointLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
#if __DEBUG__
    // TODO : DirectionalLight::IsInShadow
    return false;
#else
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
#endif
}

// set color
void PointLight::SetColor( const Color& color )
{
    _color = color;

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PointLight ), cudaMemcpyHostToDevice );
}

// set color components
void PointLight::SetColor( real32 r, real32 g, real32 b )
{
    SetColor( Color( r, g, b ) );
}

// set position
void PointLight::SetPosition( const Vector3& position )
{
    _position = position;

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PointLight ), cudaMemcpyHostToDevice );
}

// set position
void PointLight::SetPosition( real64 x, real64 y, real64 z )
{
    SetPosition( Vector3( x, y, z ) );
}

// set radiance scale
void PointLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( PointLight ), cudaMemcpyHostToDevice );
}

REX_NS_END