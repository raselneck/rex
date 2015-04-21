#define _USE_MATH_DEFINES
#include <rex/Rex.hxx>
#include <math.h>
#include <stdio.h>
#include "DeviceScene.hxx"

REX_NS_BEGIN

/// <summary>
/// Gets the next power of two that is higher than the given number.
/// </summary>
/// <param name="number">The number.</param>
static int32 GetNextPowerOfTwo( int32 number )
{
    real64 logBase2 = log( static_cast<real64>( number ) ) / log( 2.0 );
    uint32 power    = static_cast<uint32>( Math::Ceiling( logBase2 ) );

    int32 value = 1 << power;
    return value;
}

// renders the scene
void Scene::RenderToImage()
{
    // make sure the camera is up to date
    _camera.CalculateOrthonormalVectors();

    // create the host scene data
    DeviceSceneData hsd;
    hsd.Lights          = _lights;
    hsd.AmbientLight    = _ambientLight;
    hsd.Camera          = GC::DeviceAlloc<Camera>( _camera );
    hsd.Octree          = _octree;
    hsd.Image           = GC::DeviceAlloc<Image>( *_image );
    hsd.ViewPlane       = GC::DeviceAlloc<ViewPlane>( _viewPlane );
    hsd.BackgroundColor = GC::DeviceAlloc<Color>( _backgroundColor );


    // copy our image's contents over to the device
    _image->CopyHostToDevice();


    // create the device scene data (and copy from the host)
    DeviceSceneData* dsd = GC::DeviceAlloc<DeviceSceneData>( hsd );
    if ( dsd == nullptr )
    {
        return;
    }


    // prepare for the kernel
    int32 imgWidth  = GetNextPowerOfTwo( _image->GetWidth() );
    int32 imgHeight = GetNextPowerOfTwo( _image->GetHeight() );
    dim3  blocks    = dim3( 16, 16 );
    dim3  grid      = dim3( imgHeight / blocks.x + ( ( imgHeight % blocks.x ) == 0 ? 0 : 1 ),
                            imgWidth  / blocks.y + ( ( imgWidth  % blocks.y ) == 0 ? 0 : 1 ) );

    // start a timer
    Timer timer;
    timer.Start();

    // run the kernel
    SceneRenderKernel<<<grid, blocks>>>( dsd );

    // check for errors
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Render kernel failed. Reason: ", cudaGetErrorString( err ) );
        return;
    }

    // wait for the kernel to finish executing
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to synchronize device. Reason: ", cudaGetErrorString( err ) );
        return;
    }


    timer.Stop();


    // copy our image's contents back to the host
    _image->CopyDeviceToHost();



    REX_DEBUG_LOG( "Render time: ", timer.GetElapsed(), "s (~", 1 / timer.GetElapsed(), " FPS)" );
}

REX_NS_END