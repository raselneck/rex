#include "Rex.hxx"
#include "Debug.hxx"
#include <chrono>
#include <thread>

// Windows includes to open the output image
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <Windows.h>
#include <shellapi.h>


// a simple ray tracer for drawing a single red sphere
struct MultiObjectTracer : public rex::RayTracer
{
    MultiObjectTracer( rex::Scene* scene )
        : rex::RayTracer( scene )
    {
    }
    rex::Color Trace( const rex::Ray& ray ) const
    {
        rex::ShadePoint sp = _pScene->HitObjects( ray );
        if ( sp.HasHit )
        {
            return sp.Color;
        }
        return _pScene->GetBackgroundColor();
    }
};


int main( int argc, char** argv )
{
    // create the scene
    rex::Scene scene;
    scene.SetTracerType<MultiObjectTracer>();
    scene.Build( 400, 400, 0.5f );
    scene.Render();

    // save and open the image
    scene.GetImage()->Save( "output.png" );
    ShellExecute( 0, 0, TEXT( "output.png" ), 0, 0, SW_SHOW );
    std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );

    return 0;
}