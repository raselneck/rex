#include <rex/Rex.hxx>

int32 main( int32 argc, char** argv )
{
    using namespace rex;

    Scene scene;
    scene.Build( 1024, 768 );
    scene.Render();

    return 0;
}
