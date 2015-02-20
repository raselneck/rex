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


// a simple ray tracer for drawing multiple objects
struct MultiObjectTracer : public rex::Tracer
{
    MultiObjectTracer( rex::Scene* scene )
        : rex::Tracer( scene )
    {
    }
    virtual rex::Color Trace( const rex::Ray& ray ) const
    {
        return Trace( ray, 0 );
    }
    virtual rex::Color Trace( const rex::Ray& ray, int depth ) const
    {
        rex::ShadePoint sp = _pScene->HitObjects( ray );
        if ( sp.HasHit )
        {
            return sp.Color;
        }
        return _pScene->GetBackgroundColor();
    }
};


// renders a custom scene animation
void RenderSceneAnimation( rex::Scene& scene, uint32 frameCount )
{
    using namespace rex;

    // get the camera and setup some loop variables
    PerspectiveCamera* camera = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
    real64 angle = 0.0;
    uint32 imgNumber = 0;
    real64 totalTime = 0.0;
    const real64 dAngle = 360.0 / frameCount;

    // begin the loop to render!
    Timer timer;
    rex::WriteLine( "Beginning animation..." );
    while ( angle < 360.0 )
    {
        // move the camera (I know this is mixed, but I want Z to be the "major" position at angle = 0)
        real64 x = 750.0 * std::sin( angle * Math::PI_OVER_180 );
        real64 z = 750.0 * std::cos( angle * Math::PI_OVER_180 );
        camera->SetPosition( x, 0.0, z );

        rex::Write( "Rendering image ", imgNumber + 1, " / ", frameCount, "... " );

        // render the image
        timer.Start();
        scene.Render();
        timer.Stop();

        rex::WriteLine( "Done. ", timer.GetElapsed(), " seconds" );

        // save the image
        String path = "anim/img" + std::to_string( imgNumber ) + ".png";
        scene.GetImage()->Save( path.c_str() );

        ++imgNumber;
        angle += dAngle;
        totalTime += timer.GetElapsed();
    }

    rex::WriteLine();
    rex::WriteLine( "Finished rendering animation!" );
    rex::WriteLine( "Total time: ", totalTime, " seconds" );
}


int main( int argc, char** argv )
{
    using namespace rex;

    // seed the random number generator
    Random::Seed( static_cast<uint32>( time( 0 ) ) );


    // create the scene
    WriteLine( "Building scene..." );
    Scene scene;
    scene.SetSamplerType<HammersleySampler>( 4 );
    scene.SetSamplerType<RegularSampler>( 4 );
    scene.SetTracerType<MultiObjectTracer>();
    scene.SetCameraType<PerspectiveCamera>();
    {
        PerspectiveCamera* camera = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
        camera->SetPosition( 0.0, 0.0, 750.0 );
        camera->SetTarget( 0.0, 0.0, 0.0 );
        camera->SetUp( 0.0, 1.0, 0.0 );
        camera->SetViewPlaneDistance( 1000.0f );
    }
    scene.Build( 400, 400, 0.5f );


    RenderSceneAnimation( scene, 1 );
    ShellExecute( 0, 0, TEXT( "anim\\img0.png" ), 0, 0, SW_SHOW );


    rex::ReadLine();
    return 0;
}
