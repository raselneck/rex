#include "PerspectiveCamera.hxx"
#include "Color.hxx"
#include "Scene.hxx"
#include "ViewPlane.hxx"

REX_NS_BEGIN

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

    // prepare for the tracing!!
    ViewPlane    vp( scene.GetViewPlane() );
    Color        color;
    Ray          ray;
    int32        rayDepth   = 0;
    Vector2      sp; // sampler sample point
    Vector2      pp; // pixel sample point
    auto&        image      = scene.GetImage();
    auto&        sampler    = scene.GetSampler();
    auto&        tracer     = scene.GetTracer();
    const real32 invSamples = 1.0f / sampler->GetSampleCount();

    vp.PixelSize /= _zoomAmount;
    ray.Origin    = _position;

    // begin the tracing!!
    for ( int32 y = 0; y < image->GetHeight(); ++y )
    {
        for ( int32 x = 0; x < image->GetWidth(); ++x )
        {
            color = Color::Black;

            // begin the sampling!!
            for ( int32 sample = 0; sample < sampler->GetSampleCount(); ++sample )
            {
                sp = sampler->SampleUnitSquare();

                pp.X = vp.PixelSize * ( x - 0.5 * vp.Width  + sp.X );
                pp.Y = vp.PixelSize * ( y - 0.5 * vp.Height + sp.Y );

                ray.Direction = GetRayDirection( pp );
                color += tracer->Trace( ray, rayDepth );
            }

            // set the image pixel!!
            color *= invSamples;
            image->SetPixelUnchecked( x, y, color );
        }
    }
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