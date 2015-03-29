#include <rex/Graphics/Lights/LightCollection.hxx>
#include <rex/Utility/GC.hxx>
#include <rex/Utility/Logger.hxx>

REX_NS_BEGIN

// create new light collection
LightCollection::LightCollection()
    : _dLights( nullptr )
{
    // create our ambient light
    _hAmbientLight = GC::HostAlloc<AmbientLight>();

    // update our device light array
    UpdateDeviceArray();
}

// destroy this light collection
LightCollection::~LightCollection()
{
    if ( _dLights )
    {
        cudaFree( _dLights );
        _dLights = nullptr;
    }

    _hAmbientLight = nullptr;
}

// get ambient light
const AmbientLight* LightCollection::GetAmbientLight() const
{
    return _hAmbientLight;
}

// get light count
uint32 LightCollection::GetLightCount() const
{
    return static_cast<uint32>( _hLights.size() );
}

// gets the lights on the host
const std::vector<Light*>& LightCollection::GetLights() const
{
    return _hLights;
}

// get the light array on the device
const Light** LightCollection::GetDeviceLights() const
{
    // the CUDA compiler makes me do the cast :(
    return (const Light**)( _dLights );
}

// add directional light
DirectionalLight* LightCollection::AddDirectionalLight()
{
    auto light = GC::HostAlloc<DirectionalLight>();
    if ( light )
    {
        _hLights.push_back( light );
        UpdateDeviceArray();
    }
    return light;
}

// add directional light
DirectionalLight* LightCollection::AddDirectionalLight( const Vector3& direction )
{
    auto light = GC::HostAlloc<DirectionalLight>( direction );
    if ( light )
    {
        _hLights.push_back( light );
        UpdateDeviceArray();
    }
    return light;
}

// add directional light
DirectionalLight* LightCollection::AddDirectionalLight( real64 x, real64 y, real64 z )
{
    auto light = GC::HostAlloc<DirectionalLight>( x, y, z );
    if ( light )
    {
        _hLights.push_back( light );
        UpdateDeviceArray();
    }
    return light;
}

// add point light
PointLight* LightCollection::AddPointLight()
{
    auto light = GC::HostAlloc<PointLight>();
    if ( light )
    {
        _hLights.push_back( light );
        UpdateDeviceArray();
    }
    return light;
}

// add point light
PointLight* LightCollection::AddPointLight( const Vector3& position )
{
    auto light = GC::HostAlloc<PointLight>( position );
    if ( light )
    {
        _hLights.push_back( light );
        UpdateDeviceArray();
    }
    return light;
}

// add point light
PointLight* LightCollection::AddPointLight( real64 x, real64 y, real64 z )
{
    auto light = GC::HostAlloc<PointLight>( x, y, z );
    if ( light )
    {
        _hLights.push_back( light );
        UpdateDeviceArray();
    }
    return light;
}

// set ambient color
void LightCollection::SetAmbientColor( const Color& color )
{
    _hAmbientLight->SetColor( color );
}

// set ambient color
void LightCollection::SetAmbientColor( real32 r, real32 g, real32 b )
{
    _hAmbientLight->SetColor( r, g, b );
}

// update device array
void LightCollection::UpdateDeviceArray()
{
    // cleanup the old array
    if ( _dLights )
    {
        cudaFree( _dLights );
    }


    // create the array
    cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &_dLights ),
                                  _hLights.size() * sizeof( Light* ) );
    if ( err != cudaSuccess )
    {
        Logger::Log( "Failed to allocate space for light collection on device." );
        return;
    }


    // now we need to go through all of our lights and copy over their device pointers
    for ( uint32 i = 0; i < _hLights.size(); ++i )
    {
        const Light* dLight = _hLights[ i ]->GetOnDevice();
        const void*  src    = static_cast<const void*>( &dLight );
        void*        dst    = REX_OFFSET( _dLights, i * sizeof( Light* ) );

        // TODO : Make sure this memcpy is doing what I think it's doing
        err = cudaMemcpy( dst, src, sizeof( Light* ), cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            Logger::Log( "Failed to add light at index ", i, " to device array." );
        }
    }
}

REX_NS_END