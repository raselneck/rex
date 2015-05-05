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

using namespace rex;
using namespace std;

struct LaunchParameters
{
    SceneRenderMode RenderMode;
    int32 RenderWidth;
    int32 RenderHeight;
    int32 FrameCount;

    LaunchParameters()
    {
        RenderMode   = SceneRenderMode::ToOpenGL;
        RenderWidth  = 640;
        RenderHeight = 480;
        FrameCount   = 1;
    }
};

/// <summary>
/// Gets the launch parameters from the given command line arguments.
/// </summary>
/// <param name="argc">The argument count.</param>
/// <param name="argv">The argument values.</param>
LaunchParameters GetPaunchParameters( int32 argc, char** argv )
{
    LaunchParameters params;

    for ( int32 i = 0; i < argc; ++i )
    {
        // check for render mode
        if ( 0 == strcmp( argv[ i ], "--render-mode" ) && i < argc - 1 )
        {
            // OpenGL
            if ( 0 == strcmp( argv[ i + 1 ], "opengl" ) )
            {
                params.RenderMode = SceneRenderMode::ToOpenGL;
            }
            // Image
            else if ( 0 == strcmp( argv[ i + 1 ], "image" ) )
            {
                params.RenderMode = SceneRenderMode::ToImage;
            }
            i += 1;
        }
        // check for render height
        else if ( 0 == strcmp( argv[ i ], "--height" ) && i < argc - 1 )
        {
            params.RenderHeight = atoi( argv[ i + 1 ] );
            i += 1;
        }
        // check for render width
        else if ( 0 == strcmp( argv[ i ], "--width" ) && i < argc - 1 )
        {
            params.RenderWidth = atoi( argv[ i + 1 ] );

            i += 1;
        }
        // check for frame count
        else if ( 0 == strcmp( argv[ i ], "--frame-count" ) && i < argc - 1 )
        {
            params.FrameCount = atoi( argv[ i + 1 ] );
            i += 1;
        }
    }

    return params;
}

/// <summary>
/// Renders a frame.
/// </summary>
/// <param name="scene">The scene to render.</param>
/// <param name="currFrame">The current frame.</param>
/// <param name="totalFrames">The total number of frames.</param>
void RenderFrame( Scene& scene, uint32 currFrame, uint32 totalFrames )
{
    // get the camera's position
    const real32 distance = 100.0f;
    const real32 angle    = Math::TwoPi() / totalFrames * currFrame;
    real32       x        = distance * std::sin( angle );
    real32       z        = distance * std::cos( angle );

    // set the camera's position
    scene.SetCameraPosition( x, 0.0f, z );



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
bool PrintCudaDeviceInfo( int32 device )
{
    // get the device properties
    cudaDeviceProp props;
    cudaError_t    err = cudaGetDeviceProperties( &props, device );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to get device properties. Reason: ", cudaGetErrorString( err ) );
        return false;
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
    REX_DEBUG_LOG( "  Retrieving max stack size per thread..." );
    size_t stackSize = 0;
    err = cudaDeviceGetLimit( &stackSize, cudaLimitStackSize );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to get stack limit. Reason: ", cudaGetErrorString( err ) );
        return false;
    }
    REX_DEBUG_LOG( "  Max thread stack:    ", stackSize );

    // attempt to increase the limit
    const size_t desiredStackSize = 2048;
    err = cudaDeviceSetLimit( cudaLimitStackSize, desiredStackSize );
    if ( err == cudaSuccess )
    {
        REX_DEBUG_LOG( "  Changed stack size to ", desiredStackSize );
    }
    else
    {
        REX_DEBUG_LOG( "  Failed to increase stack size to ", desiredStackSize );
        return false;
    }

    return true;
}

/// <summary>
/// Runs a scene that renders to an OpenGL window.
/// </summary>
void RunOpenGLScene( const LaunchParameters& params )
{
    Scene scene( SceneRenderMode::ToOpenGL );
    if ( scene.Build( params.RenderWidth, params.RenderHeight ) )
    {
        scene.Render();
    }
}

/// <summary>
/// Runs a scene that renders to a series of images.
/// </summary>
/// <param name="frameCount">The total number of frames.</param>
void RunImageScene( const LaunchParameters& params )
{
    Scene scene( SceneRenderMode::ToImage );
    if ( scene.Build( params.RenderWidth, params.RenderHeight ) )
    {
        // create our output directory
        mkdir( "render" );

        // render all our frames
        uint32 uFrameCount = static_cast<uint32>( params.FrameCount );
        for ( uint32 i = 0; i < uFrameCount; ++i )
        {
            RenderFrame( scene, i, uFrameCount );
        }

#if defined( _WIN32 ) || defined( _WIN64 )
        // open the first image
        ShellExecuteA( 0, 0, "render\\img0.png", 0, 0, SW_SHOW );
#endif
    }
}

/// <summary>
/// The program entry point.
/// </summary>
/// <param name="argc">The argument count.</param>
/// <param name="argv">The argument values.</param>
int32 main( int32 argc, char** argv )
{
    // ensure we can configure the CUDA device
    if ( !PrintCudaDeviceInfo( 0 ) )
    {
        return -1;
    }

    // get the launch parameters and ensure the width and height are okay
    LaunchParameters params = GetPaunchParameters( argc, argv );
    if ( params.RenderHeight < 1 || params.RenderWidth < 1 )
    {
        REX_DEBUG_LOG( "ERROR: Cannot render with dimensions less than 1x1." );
        REX_DEBUG_LOG( "Given dimensions: ", params.RenderWidth, "x", params.RenderHeight );
        return -1;
    }

    // run the scene
    if ( params.RenderMode == SceneRenderMode::ToOpenGL )
    {
        RunOpenGLScene( params );
    }
    else if ( params.RenderMode == SceneRenderMode::ToImage )
    {
        RunImageScene( params );
    }

    return 0;
}
