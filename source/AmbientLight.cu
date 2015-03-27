#include <rex/Graphics/Lights/AmbientLight.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create ambient light
AmbientLight::AmbientLight()
    : _radianceScale( 1.0f ),
      _color( Color::White() ),
      _dThis( nullptr )
{
    _castShadows = false;

    // create us on the device
    _dThis = GC::DeviceAlloc<AmbientLight>( *this );
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
__device__ Vector3 AmbientLight::GetLightDirection( ShadePoint& sp )
{
    return Vector3( 0.0 );
}

// get radiance
__device__ Color AmbientLight::GetRadiance( ShadePoint& sp )
{
    return _radianceScale * _color;
}

// get type
LightType AmbientLight::GetType() const
{
    return LightType::AmbientLight;
}

// check if in shadow
__device__ bool AmbientLight::IsInShadow( const Ray& ray, const ShadePoint& sp ) const
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

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( AmbientLight ), cudaMemcpyHostToDevice );
}

// set color by components
void AmbientLight::SetColor( real32 r, real32 g, real32 b )
{
    SetColor( Color( r, g, b ) );
}

// set radiance scale
void AmbientLight::SetRadianceScale( real32 ls )
{
    _radianceScale = ls;

    // update us on the device
    cudaMemcpy( _dThis, this, sizeof( AmbientLight ), cudaMemcpyHostToDevice );
}

REX_NS_END