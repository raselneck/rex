#include "Rex.hxx"
#include "Debug.hxx"

// Windows includes to open the output image
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <Windows.h>
#include <shellapi.h>


// a simple ray tracer for drawing a single red sphere
struct RedSphereRayTracer : public rex::RayTracer
{
    RedSphereRayTracer( rex::Scene* scene )
        : rex::RayTracer( scene )
    {
    }

    rex::Color Trace( const rex::Ray& ray ) const
    {
        rex::ShadePoint sp( _scenePtr );
        real64 t;

        if ( _scenePtr->GetSphere().Hit( ray, t, sp ) )
        {
            return rex::Color::Red;
        }
        return rex::Color::Black;
    }
};


int main( int argc, char** argv )
{
    // create the scene
    rex::Scene scene;
    scene.SetTracerType<RedSphereRayTracer>();
    scene.Build( 800, 600 );
    scene.Render();


    // save and open the image
    scene.GetImage()->Save( "output.png" );
    ShellExecute( 0, 0, TEXT( "output.png" ), 0, 0, SW_SHOW );


    // wait for key press (and give time for the shell execute)
    std::string temp;
    rex::WriteLine( "Press any key to continue. . ." );
    std::getline( std::cin, temp );

    return 0;
}