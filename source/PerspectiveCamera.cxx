#include <rex/Cameras/PerspectiveCamera.hxx>
#include <rex/Utility/Color.hxx>
#include <rex/Scene/Scene.hxx>
#include <rex/Scene/ViewPlane.hxx>
#include <thread>

REX_NS_BEGIN

// the render callback for threads
static void ThreadRenderCallback( PerspectiveCamera& camera, Scene& scene, ViewPlane& vp, int32 startX, int32 startY, int32 endX, int32 endY )
{
    // prepare for the tracing!!
    Color        color;
    Ray          ray;
    int32        rayDepth = 0;
    Vector2      sp; // sampler sample point
    Vector2      pp; // pixel sample point
    auto&        image = scene.GetImage();
    auto&        sampler = scene.GetSampler();
    auto&        tracer = scene.GetTracer();
    const real32 invSamples = 1.0f / sampler->GetSampleCount();

    ray.Origin = camera.GetPosition();

    // begin the tracing!!
    for ( int32 y = startY; y < endY; ++y )
    {
        for ( int32 x = startX; x < endX; ++x )
        {
            color = Color::Black;

#if 0
            // NOTE : Using the sampler is broken when multi-threaded

            // begin the sampling!!
            for ( int32 sample = 0; sample < sampler->GetSampleCount(); ++sample )
            {
                sp = sampler->SampleUnitSquare();
            
                pp.X = vp.PixelSize * ( x - 0.5 * vp.Width  + sp.X );
                pp.Y = vp.PixelSize * ( y - 0.5 * vp.Height + sp.Y );
            
                ray.Direction = camera.GetRayDirection( pp );
                color += tracer->Trace( ray, rayDepth );
            }
#endif

#if 1
            int32  n    = static_cast<int32>( sqrt( sampler->GetSampleCount() ) );
            real64 invn = 1.0 / n;
            for ( int32 sy = 0; sy < n; ++sy )
            {
                for ( int32 sx = 0; sx < n; ++sx )
                {
                    pp.X = vp.PixelSize * ( x - 0.5 * vp.Width  + ( sx + 0.5 ) * invn );
                    pp.Y = vp.PixelSize * ( y - 0.5 * vp.Height + ( sy + 0.5 ) * invn );

                    ray.Direction = camera.GetRayDirection( pp );
                    color += tracer->Trace( ray, rayDepth );
                }
            }
#endif

            // set the image pixel!!
            color *= invSamples;
            image->SetPixelUnchecked( x, y, color );
        }
    }
}

// new perspective camera
PerspectiveCamera::PerspectiveCamera()
    : _viewPlaneDistance( 1000.0f ), _zoomAmount( 1.0f )
{
}

// destroy perspective camera
PerspectiveCamera::~PerspectiveCamera()
{
    _viewPlaneDistance = 0.0f;
    _zoomAmount        = 0.0f;
}

// get ray direction to point
Vector3 PerspectiveCamera::GetRayDirection( const Vector2& sp ) const
{
    Vector3 dir = sp.X * _orthoU                // +x is right
                - sp.Y * _orthoV                // +y is up
                - _viewPlaneDistance * _orthoW; // +z is out of screen
    return Vector3::Normalize( dir );
}

// render scene
void PerspectiveCamera::Render( Scene& scene )
{
    CalculateUVW();

    ViewPlane    vp( scene.GetViewPlane() );
    vp.PixelSize /= _zoomAmount;

    auto& image = scene.GetImage();

    // original, single-threaded rendering
#if 0
    std::thread t1( ThreadRenderCallback,
                    std::ref( *this ),
                    std::ref( scene ),
                    std::ref( vp    ),
                    0, 0, image->GetWidth(), image->GetHeight() );
    t1.join();
#endif

    // four threads (four seems to be the magic number for speed)
#if 1
    int32 farX = image->GetWidth();
    int32 farY = image->GetHeight();
    int32 midX = farX / 2;
    int32 midY = farY / 2;

    auto refThis      = std::ref( *this );
    auto refScene     = std::ref( scene );
    auto refViewPlane = std::ref( vp );

    // create our four threads
    std::thread t1( ThreadRenderCallback, refThis, refScene, refViewPlane, 0,    0,    midX, midY );
    std::thread t2( ThreadRenderCallback, refThis, refScene, refViewPlane, midX, 0,    farX, midY );
    std::thread t3( ThreadRenderCallback, refThis, refScene, refViewPlane, 0,    midY, midX, farY );
    std::thread t4( ThreadRenderCallback, refThis, refScene, refViewPlane, midX, midY, farX, farY );

    // now wait for the threads to finish
    t1.join();
    t2.join();
    t3.join();
    t4.join();
#endif
}

// set view distance
void PerspectiveCamera::SetViewPlaneDistance( real32 dist )
{
    _viewPlaneDistance = dist;
}

// set zoom amount
void PerspectiveCamera::SetZoomAmount( real32 zoom )
{
    _zoomAmount = zoom;
}

REX_NS_END