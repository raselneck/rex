#include <rex/Rex.hxx>

REX_NS_BEGIN

/// <summary>
/// Defines a set of scene cleanup data.
/// </summary>
struct SceneDisposeData
{
    DeviceList<Light*>*    Lights;
    AmbientLight*          AmbientLight;
    DeviceList<Geometry*>* Geometry;
    Octree*                Octree;
};

/// <summary>
/// The scene dispose kernel.
/// </summary>
/// <param name="data">The data to dispose.</param>
__global__ void SceneDisposeKernel( SceneDisposeData* data )
{
    // delete the geometry
    if ( data->Geometry )
    {
        for ( uint32 i = 0; i < data->Geometry->GetSize(); ++i )
        {
            Geometry* geom = data->Geometry->operator[]( i );
            delete    geom;
        }
        delete data->Geometry;
    }

    // delete the lights
    if ( data->Lights )
    {
        for ( uint32 i = 0; i < data->Lights->GetSize(); ++i )
        {
            Light* light = data->Lights->operator[]( i );
            delete light;
        }
        delete data->Lights;
    }

    // delete the ambient light
    if ( data->AmbientLight )
    {
        delete data->AmbientLight;
    }

    // delete the octree
    if ( data->Octree )
    {
        delete data->Octree;
    }
}

// dispose of the scene
void Scene::Dispose()
{
    REX_DEBUG_LOG( "Disposing of scene..." );


    // delete the OpenGL texture
    if ( _texture )
    {
        delete _texture;
        _texture = nullptr;
    }

    // delete the OpenGL window
    if ( _window )
    {
        delete _window;
        _window = nullptr;
    }

    // delete the image
    if ( _image )
    {
        delete _image;
        _image = nullptr;
    }



    // prepare to call the dispose kernel
    SceneDisposeData  sdHost = { _lights, _ambientLight, _geometry, _octree };
    SceneDisposeData* sdDevice = nullptr;

    // allocate and copy the cleanup information
    if ( cudaSuccess != cudaMalloc( (void**)( &sdDevice ), sizeof( SceneDisposeData ) ) )
    {
        REX_DEBUG_LOG( "  Failed to allocate space for data." );
        return;
    }
    if ( cudaSuccess != cudaMemcpy( sdDevice, &sdHost, sizeof( SceneDisposeData ), cudaMemcpyHostToDevice ) )
    {
        REX_DEBUG_LOG( "  Failed to initialize device data." );
        return;
    }

    // call the kernel
    SceneDisposeKernel<<<1, 1>>>( sdDevice );

    // check for errors
    if ( cudaSuccess != cudaGetLastError() )
    {
        REX_DEBUG_LOG( "  Scene disposal failed. Reason: ", cudaGetErrorString( cudaGetLastError() ) );
        return;
    }

    // wait for the kernel to finish executing
    if ( cudaSuccess != cudaDeviceSynchronize() )
    {
        REX_DEBUG_LOG( "  Failed to synchronize device. Reason: ", cudaGetErrorString( cudaDeviceSynchronize() ) );
        return;
    }

    // now set everything to null :D
    _lights         = nullptr;
    _ambientLight   = nullptr;
    _geometry       = nullptr;
    _octree         = nullptr;


    // try to reset the device
    if ( cudaSuccess != cudaDeviceReset() )
    {
        REX_DEBUG_LOG( "  Failed to reset device." );
    }
}

REX_NS_END