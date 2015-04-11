#include <rex/Rex.hxx>

using namespace rex;

int32 main( int32 argc, char** argv )
{
    Scene scene;

    Logger::Log( "Building scene..." );
    scene.Build( 1024, 768 );

    Logger::Log( "Rendering scene..." );
    scene.Render();

#if defined( _WIN32 ) || defined( _WIN64 )
    system( "pause" );
#endif

    return 0;
}
