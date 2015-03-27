#include <rex/Graphics/Lights/DirectionalLight.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

// TODO : Override SetCastsShadows to update on device

REX_NS_BEGIN

// create light
DirectionalLight::DirectionalLight()
    : DirectionalLight( Vector3() )
{
}

// create light w/ direction components
DirectionalLight::DirectionalLight( real64 x, real64 y, real64 z )
    : DirectionalLight( Vector3( x, y, z ) )
{
}

// create light w/ direction
DirectionalLight::DirectionalLight( const Vector3& direction )
    : _direction( Vector3::Normalize( direction ) ),
      _color( Color::White() ),
      _radianceScale( 1.0f ),
      _dThis( nullptr )
{
    _castShadows = true;

    // create us on the device
    _dThis = GC::DeviceAlloc<DirectionalLight>( *this );
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
__device__ Vector3 DirectionalLight::GetLightDirection( ShadePoint& sp )
{
    return _direction;
}

// get light on the device
Light* DirectionalLight::GetOnDevice()
{
    return static_cast<Light*>( _dThis );
}

// get radiance
__device__ Color DirectionalLight::GetRadiance( ShadePoint& sp )
{
    return _radianceScale * _color;
}

// get type
LightType DirectionalLight::GetType() const
{
    return LightType::DirectionalLight;
}

// check if in shadow
__device__ bool DirectionalLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
{
    // I'm guessing at this implementation, as Suffern does not provide one.
    // it seems to work, so if the glove fits...

    Ray ray2( ray.Origin, _direction );
    bool inShadow = sp.Scene->ShadowHitObjects( ray2 );
    return inShadow;
}

// set color
void DirectionalLight::SetColor( const Color& color )
{
    _color = color;

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( DirectionalLight ), cudaMemcpyHostToDevice );
}

// set color w/ components
void DirectionalLight::SetColor( real32 r, real32 g, real32 b )
{
    SetColor( Color( r, g, b ) );
}

// set direction
void DirectionalLight::SetDirection( const Vector3& direction )
{
    _direction = Vector3::Normalize( direction );

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( DirectionalLight ), cudaMemcpyHostToDevice );
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

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( DirectionalLight ), cudaMemcpyHostToDevice );
}

REX_NS_END