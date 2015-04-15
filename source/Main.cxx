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
/// The program entry point.
/// </summary>
/// <param name="argc">The argument count.</param>
/// <param name="argv">The argument values.</param>
int32 main( int32 argc, char** argv )
{
    Scene scene;
    if ( scene.Build( 1024, 768 ) )
    {
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
    }

    // wait for user input
    system( "pause" );
#else
    }
#endif

    return 0;
}
