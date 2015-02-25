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
    const real64       eyeHeight  = 250.0;
    const real64       eyeCorrect = 10.0 * ( 2.0 / 3.0 );


    // make the output directory
    mkdir( "anim" );


    // build scene octree
    rex::Write( "Building scene octree... " );
    scene.BuildOctree();
    rex::WriteLine( "Done." );


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

// renders a sphere animation
void RenderSphereAnimation( rex::Scene& scene, uint32 frameCount, real64 eyeHeight, real64 eyeDist )
{
    using namespace rex;

    const real64 dAngle = 2.0 * Math::INV_PI;
    real64       angle = 0.0;
    mkdir( "anim" );

    // camera
    auto* camera = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
    camera->SetPosition( 0.0, eyeHeight * ( 20.0 / 3.0 ), eyeDist );
    camera->SetTarget  ( 0.0, eyeHeight / ( 20.0 / 3.0 ), 0.0 );


    // materials
    const real32 ka = 0.25f;
    const real32 kd = 0.75f;
    auto material = MatteMaterial( Color::White, ka, kd );


    // sphere
    auto sphere = scene.AddSphere( Vector3(), 15.0, material );


    // render loop
    Timer timer;
    for ( uint32 frame = 0; frame < frameCount; ++frame )
    {
        angle += dAngle;

        // set the sphere's position
        real64 y = std::sin( angle ) * 20.0 + 20.0;
        sphere->SetCenter( 0.0, y, 0.0 );

        // render the scene
        timer.Start();
        scene.Render();
        timer.Stop();

        rex::Write( "Rendering frame ", frame + 1, " / ", frameCount, "... " );

        // save the image
        String path = "anim/img" + std::to_string( frame ) + ".png";
        scene.GetImage()->Save( path.c_str() );

        // print out how long it took
        rex::WriteLine( "Done. (", timer.GetElapsed(), " seconds)" );
    }
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
    scene.SetSamplerType<RegularSampler>( 9 );
    scene.SetCameraType<PerspectiveCamera>();
    {
        PerspectiveCamera* camera = reinterpret_cast<PerspectiveCamera*>( scene.GetCamera().get() );
        camera->SetTarget( 0.0, 0.0, 0.0  );
        camera->SetViewPlaneDistance( 1750.0f );
    }
    scene.Build( 1280, 720, 0.5f );


    //RenderSphereAnimation( scene, 1, 100.0, 450.0 );
    RenderSceneAnimation( scene, 1, 750.0 );
#if defined( _WIN32 ) || defined( _WIN64 )
    ShellExecute( 0, 0, TEXT( "anim\\img0.png" ), 0, 0, SW_SHOW );
#endif


    rex::ReadLine();
    return 0;
}
