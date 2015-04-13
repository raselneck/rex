#include <rex/Graphics/Lights/Light.hxx>

REX_NS_BEGIN

// create light
__device__ Light::Light( LightType type )
    : _castShadows( false ),
      _type( type )
{
}

// destroy light
__device__ Light::~Light()
{
    _castShadows = 0;
}

// check if casts shadows
__device__ bool Light::CastsShadows() const
{
    return _castShadows;
}

// get light type
__device__ LightType Light::GetType() const
{
    return _type;
}

// set whether to cast shadows
__device__ void Light::SetCastShadows( bool value )
{
    _castShadows = value;
}

REX_NS_END