#include <rex/Rex.hxx>
#include <sstream>
#if defined( _WIN32 ) || defined( _WIN64 )
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  define NOMINMAX
#  include <Windows.h>
#  include <shellapi.h>
#  include <direct.h>
#  define mkdir _mkdir
#else
#  include <sys/stat.h>
#  define mkdir(path) mkdir(path, S_IRWXU)
#endif

// Older OpenGL interop
// http://rauwendaal.net/2011/02/10/how-to-use-cuda-3-0s-new-graphics-interoperability-api-with-opengl/
// Non-deprecated OpenGL interop
// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__OPENGL.html
// Graphics interop
// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__INTEROP.html

using namespace rex;
using namespace std;

/// <summary>
/// Renders a frame.
/// </summary>
/// <param name="scene">The scene to render.</param>
/// <param name="currFrame">The current frame.</param>
/// <param name="totalFrames">The total number of frames.</param>
void RenderFrame( Scene& scene, uint32 currFrame, uint32 totalFrames )
{
    // get the camera's position
    const real_t distance = real_t( 100.0 );
    const real_t angle    = Math::TwoPi() / totalFrames * currFrame;
    real_t       x        = distance * std::sin( angle );
    real_t       z        = distance * std::cos( angle );

    // set the camera's position
    scene.SetCameraPosition( x, real_t( 0.0 ), z );



    // render the scene, save the image, and release the memory the scene used
    scene.Render();
    {
        // get image name
        ostringstream stream;
        stream << "render\\img" << currFrame << ".png";
        string fname = stream.str();

        // save the image
        scene.SaveImage( fname.c_str() );
    }
    GC::ReleaseDeviceMemory();
}

/// <summary>
/// Prints information about the given CUDA device.
/// </summary>
/// <param name="device">The device number to get the information about.</param>
void PrintCudaDeviceInfo( int32 device )
{
    // get the device properties
    cudaDeviceProp props;
    cudaError_t    err = cudaGetDeviceProperties( &props, device );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to get device properties. Reason: ", cudaGetErrorString( err ) );
        return;
    }

    // print out the device properties
    REX_DEBUG_LOG( "CUDA Device Properties:" );
    REX_DEBUG_LOG( "  Name:                ", props.name );
    REX_DEBUG_LOG( "  Compute capability:  ", props.major, ".", props.minor );
    REX_DEBUG_LOG( "  Clock rate:          ", props.clockRate, " KHz (", props.clockRate / 1024.0f / 1024.0f, " GHz)" );
    REX_DEBUG_LOG( "  Global memory:       ", props.totalGlobalMem, " (", props.totalGlobalMem / 1024.0f / 1024.0f, " GB)" );
    REX_DEBUG_LOG( "  Multiprocessors:     ", props.multiProcessorCount );
    REX_DEBUG_LOG( "  Warp size:           ", props.warpSize );
    REX_DEBUG_LOG( "  Max threads / block: ", props.maxThreadsPerBlock );
    REX_DEBUG_LOG( "  Max dimension size:  ", props.maxThreadsDim[ 0 ], ",", props.maxThreadsDim[ 1 ], ",", props.maxThreadsDim[ 2 ] );
    REX_DEBUG_LOG( "  Max grid size:       ", props.maxGridSize[ 0 ], ",", props.maxGridSize[ 1 ], ",", props.maxGridSize[ 2 ] );
    REX_DEBUG_LOG( "  Max threads / proc.: ", props.maxThreadsPerMultiProcessor );

    // get each thread's stack size
    size_t stackSize = 0;
    cudaDeviceGetLimit( &stackSize, cudaLimitStackSize );
    REX_DEBUG_LOG( "  Max thread stack:    ", stackSize );
}

/// <summary>
/// The program entry point.
/// </summary>
/// <param name="argc">The argument count.</param>
/// <param name="argv">The argument values.</param>
int32 main( int32 argc, char** argv )
{
    PrintCudaDeviceInfo( 0 );

    Scene scene( SceneRenderMode::ToImage );
    if ( scene.Build( 1024, 768 ) )
    {
        //scene.Render();

        // create our output directory
        mkdir( "render" );

        // render all our frames
        const uint32 totalFrameCount = 1;
        for ( uint32 i = 0; i < totalFrameCount; ++i )
        {
            RenderFrame( scene, i, totalFrameCount );
        }

#if defined( _WIN32 ) || defined( _WIN64 )
        // open the first image
        ShellExecuteA( 0, 0, "render\\img0.png", 0, 0, SW_SHOW );
#endif
        //*/
    }

    return 0;
}
