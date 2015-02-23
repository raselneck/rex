#include <rex/Rex.hxx>
#include <rex/Debug.hxx>
#include <chrono>
#include <cmath>
#include <thread>

// Windows includes to open the output image
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


// the main ray-cast tracer
class RayCastTracer : public rex::Tracer
{
    mutable rex::ShadePoint _sp;

public:
    RayCastTracer( rex::Scene* scene )
        : rex::Tracer( scene ), _sp( scene )
    {
    }

    virtual rex::Color Trace( const rex::Ray& ray ) const
    {
        return Trace( ray, 0 );
    }

    virtual rex::Color Trace( const rex::Ray& ray, int32 depth ) const
    {
        _sp.Reset();
        
        _scene->HitObjects( ray, _sp );
        if ( _sp.HasHit )
        {
            _sp.Ray = ray;
            return _sp.Material->Shade( _sp );
        }

        return _scene->GetBackgroundColor();
    }
};


// renders a custom scene animation
void RenderSceneAnimation( rex::Scene& scene, uint32 frameCount, real64 dist )
{
    using namespace rex;

    // get the camera and setup some loop variables
    PerspectiveCamera* camera     = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
    real64             angle      = 0.0;
    uint32             imgNumber  = 0;
    real64             totalTime  = 0.0;
    const real64       dAngle     = 360.0 / frameCount;
    const real64       eyeHeight  = 50.0;
    const real64       eyeCorrect = 10.0 * ( 2.0 / 3.0 );

    // make the output directory
    mkdir( "anim" );

    // begin the loop to render!
    Timer timer;
    camera->SetTarget( 0.0, eyeHeight / eyeCorrect, 0.0 );
    rex::WriteLine( "Beginning animation...\n" );
    while ( angle < 360.0 )
    {
        // move the camera (I know this is mixed, but I want Z to be the "major" position at angle = 0)
        real64 x = dist * std::sin( angle * Math::PI_OVER_180 );
        real64 z = dist * std::cos( angle * Math::PI_OVER_180 );
        camera->SetPosition( x, eyeHeight, z );

        
        rex::Write( "Rendering image ", imgNumber + 1, " / ", frameCount, "... " );

        // render the image
        timer.Start();
        scene.Render();
        timer.Stop();

        rex::WriteLine( "Done. (", timer.GetElapsed(), " seconds)" );

        // save the image
        String path = "anim/img" + std::to_string( imgNumber ) + ".png";
        scene.GetImage()->Save( path.c_str() );

        ++imgNumber;
        angle += dAngle;
        totalTime += timer.GetElapsed();
    }

    rex::WriteLine();
    rex::WriteLine( "Finished rendering animation!" );
    rex::WriteLine( "* Sample count:  ", scene.GetSampler()->GetSampleCount() );
    rex::WriteLine( "* Light count:   ", scene.GetLightCount() );
    rex::WriteLine( "* Object count:  ", scene.GetObjectCount() );
    rex::WriteLine( "* Average time:  ", totalTime / frameCount, " seconds / frame" );
    rex::WriteLine( "* Total time:    ", totalTime, " seconds" );
}


int main( int argc, char** argv )
{
    using namespace rex;

    // seed the random number generator
    Random::Seed( static_cast<uint32>( time( 0 ) ) );


    // create the scene
    rex::WriteLine( "Building scene..." );
    Scene scene;
    scene.SetTracerType<RayCastTracer>();
    scene.SetSamplerType<RegularSampler>( 4 );
    scene.SetCameraType<PerspectiveCamera>();
    {
        PerspectiveCamera* camera = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
        camera->SetTarget( 0.0, 0.0, 0.0  );
        camera->SetUp    ( 0.0, 1.0, 0.0  );
        camera->SetViewPlaneDistance( 1750.0f );
    }
    scene.Build( 1280, 720, 0.5f );


    RenderSceneAnimation( scene, 1, 750.0 );
#if defined( _WIN32 ) || defined( _WIN64 )
    ShellExecute( 0, 0, TEXT( "anim\\img0.png" ), 0, 0, SW_SHOW );
#endif


    rex::ReadLine();
    return 0;
}
