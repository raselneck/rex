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
    const real64       eyeHeight  = 250.0;


    // make the output directory
    mkdir( "anim" );


    // build scene octree
    rex::Write( "Building scene octree... " );
    scene.BuildOctree();
    rex::WriteLine( "Done." );


    // begin the loop to render!
    Timer timer;
    camera->SetTarget( 0.0, 0.0, 0.0 );
    rex::WriteLine( "Beginning animation...\n" );
    while ( angle < 360.0 )
    {
        // move the camera (I know this is mixed, but I want Z to be the "major" position at angle = 0)
        real64 x = dist * std::sin( angle * Math::PI_OVER_180 );
        real64 z = dist * std::cos( angle * Math::PI_OVER_180 );
        Vector3 position( x, 0.0, z );
        camera->SetPosition( position );
        camera->SetTarget( position + Vector3( 0.0, 0.0, -1.0 ) );

        
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
    //const uint32 seed = static_cast<uint32>( time( 0 ) );
    const uint32 seed = 1337U;
    Random::Seed( seed );
    rex::WriteLine( "Random seed: ", seed );


    // create the scene
    rex::WriteLine( "Building scene..." );
    Scene scene;
    scene.SetTracerType<RayCastTracer>();
    scene.SetSamplerType<RegularSampler>( 4 );
    scene.SetCameraType<PerspectiveCamera>();
    {
        PerspectiveCamera* camera = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
        camera->SetTarget( 0.0, 0.0, 0.0  );
        camera->SetViewPlaneDistance( 2000.0f );
    }
    scene.Build( 1280, 720, 0.5f );


    RenderSceneAnimation( scene, 1, 2000.0 );
#if defined( _WIN32 ) || defined( _WIN64 )
    ShellExecute( 0, 0, TEXT( "anim\\img0.png" ), 0, 0, SW_SHOW );
#endif


    rex::ReadLine();
    return 0;
}
