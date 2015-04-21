#include <rex/Rex.hxx>

REX_NS_BEGIN

/// <summary>
/// Defines a set of scene cleanup data.
/// </summary>
struct SceneCleanupData
{
    DeviceList<Light*>*    Lights;
    AmbientLight*          AmbientLight;
    DeviceList<Geometry*>* Geometry;
    Octree*                Octree;
};


/// <summary>
/// The scene cleanup kernel.
/// </summary>
/// <param name="data">The data to cleanup.</param>
__global__ void SceneCleanupKernel( SceneCleanupData* data )
{
    if ( data->Geometry )
    {
        for ( uint_t i = 0; i < data->Geometry->GetSize(); ++i )
        {
            Geometry* geom = data->Geometry->operator[]( i );
            delete    geom;
        }
        delete data->Geometry;
    }

    if ( data->Lights )
    {
        for ( uint_t i = 0; i < data->Lights->GetSize(); ++i )
        {
            Light* light = data->Lights->operator[]( i );
            delete light;
        }
        delete data->Lights;
    }

    if ( data->AmbientLight )
    {
        delete data->AmbientLight;
    }

    if ( data->Octree )
    {
        delete data->Octree;
    }
}


// create a new scene
Scene::Scene()
    : _lights  ( nullptr ),
      _geometry( nullptr ),
      _octree  ( nullptr ),
      _texture ( nullptr )
{
}

// destroy this scene
Scene::~Scene()
{
    REX_DEBUG_LOG( "Cleaning up scene..." );


    // delete the OpenGL texture
    if ( _texture )
    {
        delete _texture;
        _texture = nullptr;
    }



    // prepare to call the cleanup kernel
    SceneCleanupData  sdHost = { _lights, _ambientLight, _geometry, _octree };
    SceneCleanupData* sdDevice = nullptr;

    // allocate and copy the cleanup information
    if ( cudaSuccess != cudaMalloc( (void**)( &sdDevice ), sizeof( SceneCleanupData ) ) )
    {
        REX_DEBUG_LOG( "  Failed to allocate space for data." );
        return;
    }
    if ( cudaSuccess != cudaMemcpy( sdDevice, &sdHost, sizeof( SceneCleanupData ), cudaMemcpyHostToDevice ) )
    {
        REX_DEBUG_LOG( "  Failed to initialize device data." );
        return;
    }

    // call the kernel
    SceneCleanupKernel<<<1, 1>>>( sdDevice );

    // check for errors
    if ( cudaSuccess != cudaGetLastError() )
    {
        REX_DEBUG_LOG( "  Scene cleanup failed. Reason: ", cudaGetErrorString( cudaGetLastError() ) );
        return;
    }

    // wait for the kernel to finish executing
    if ( cudaSuccess != cudaDeviceSynchronize() )
    {
        REX_DEBUG_LOG( "  Failed to synchronize device. Reason: ", cudaGetErrorString( cudaDeviceSynchronize() ) );
        return;
    }

    // now set everything to null :D
    _lights       = nullptr;
    _ambientLight = nullptr;
    _geometry     = nullptr;
    _octree       = nullptr;


    // try to reset the device
    if ( cudaSuccess != cudaDeviceReset() )
    {
        REX_DEBUG_LOG( "  Failed to reset device." );
    }
}

// saves this scene's image
void Scene::SaveImage( const char* fname ) const
{
    if ( _image )
    {
        _image->Save( fname );
    }
}

// set camera position
void Scene::SetCameraPosition( const Vector3& pos )
{
    _camera.SetPosition( pos );
}

// set camera position
void Scene::SetCameraPosition( real_t x, real_t y, real_t z )
{
    _camera.SetPosition( x, y, z );
}

REX_NS_END