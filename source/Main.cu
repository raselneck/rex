#include <rex/Rex.hxx>
#if defined( _WIN32 ) || defined( _WIN64 )
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  define NOMINMAX
#  include <Windows.h>
#  include <shellapi.h>
#endif

// TODO : Need to pass in a ShadePoint to Octree hit queries to be more accurate

using namespace rex;

int32 main( int32 argc, char** argv )
{
    const char* fname = "render.png";

    Scene scene;
    scene.Build( 1024, 768 );
    scene.Render();
    scene.SaveImage( fname );

#if defined( _WIN32 ) || defined( _WIN64 )
    ShellExecuteA( 0, 0, fname, 0, 0, SW_SHOW );
#endif

    return 0;
}
