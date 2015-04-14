#define _USE_MATH_DEFINES
#include <rex/Rex.hxx>
#include <math.h>
#include <stdio.h>

REX_NS_BEGIN

/// <summary>
/// Contains scene data destined for a device.
/// </summary>
struct DeviceSceneData
{
    const DeviceList<Light*>* Lights;
    const AmbientLight*       AmbientLight;
    const Camera*             Camera;
    const Octree*             Octree;
    Image*                    Image;
    const ViewPlane*          ViewPlane;
    const Color*              BackgroundColor;
};

/// <summary>
/// Gets the next power of two that is higher than the given number.
/// </summary>
/// <param name="number">The number.</param>
static int32 GetNextPowerOfTwo( int32 number )
{
    real64 logBase2 = log( static_cast<real64>( number ) ) / log( 2.0 );
    uint32 power    = static_cast<uint32>( Math::Ceiling( logBase2 ) );

    int32 value = 1 << power;
    return value;
}

/// <summary>
/// Shadow hits objects in a scene.
/// </summary>
/// <param name="sd">The scene data.</param>
/// <param name="ray">The ray to check.</param>
/// <param name="sp">The shade point whose data should be populated.</param>
__device__ void SceneHitObjects( DeviceSceneData* sd, const Ray& ray, ShadePoint& sp )
{
    // prepare to check objects
    real_t t = 0.0;


    // only get the objects that the ray hits
    const Octree*   octree = sd->Octree;
    const Geometry* geom   = octree->QueryIntersections( ray, t );


    // iterate through the hit objects
    if ( geom && geom->Hit( ray, t, sp ) )
    {
        sp.HasHit   = true;
        sp.Ray      = ray;
        sp.Material = geom->GetMaterial();
        sp.HitPoint = ray.Origin + t * ray.Direction;
        sp.T        = t;
    }
}

/// <summary>
/// The scene render kernel.
/// </summary>
/// <param name="sd">The scene data.</param>
__global__ void SceneRenderKernel( DeviceSceneData* sd )
{
    // prepare for the tracing!!
    Ray              ray;
    Vector2          pp; // pixel sample point
    Color            color      = Color::Black();
    Image*           image      = sd->Image;
    const Camera*    camera     = sd->Camera;
    const ViewPlane* vp         = sd->ViewPlane;
    const real_t     invSamples = 1.0f / sd->ViewPlane->SampleCount;
    ShadePoint       sp         = nullptr;

    // set the ray's origin
    ray.Origin = camera->GetPosition();

    // get the image coordinates
    //int32 x = threadIdx.x;
    //int32 y = threadIdx.y;
    int32 x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    int32 y = ( blockIdx.y * blockDim.y ) + threadIdx.y;

    if ( x >= image->GetWidth() || y >= image->GetHeight() )
    {
        return;
    }

    // SAMPLE
    int32  n    = static_cast<int32>( sqrtf( sd->ViewPlane->SampleCount ) );
    real_t invn = 1.0 / n;
    for ( int32 sy = 0; sy < n; ++sy )
    {
        for ( int32 sx = 0; sx < n; ++sx )
        {
            // get the pixel point
            pp.X = x - real_t( 0.5 ) * vp->Width  + ( sx + real_t( 0.5 ) ) * invn;
            pp.Y = y - real_t( 0.5 ) * vp->Height + ( sy + real_t( 0.5 ) ) * invn;

            // set the ray direction
            ray.Direction = camera->GetRayDirection( pp );

            // hit the objects in the scene
            SceneHitObjects( sd, ray, sp );

            // add to the color if the ray hit
            if ( sp.HasHit )
            {
                sp.Ray = ray;

                const Material* mat = sp.Material;
                MaterialType    matType = mat->GetType();

                color += mat->Shade( sp, sd->Lights, sd->Octree );
            }
            else
            {
                color += *( sd->BackgroundColor );
            }
        }
    }


    // set the image pixel!
    color *= invSamples;
    image->SetDevicePixel( x, y, color );
}

// renders the scene
void Scene::Render()
{
    Logger::Log( "Rendering scene..." );



    // create the host scene data
    DeviceSceneData hsd;
    hsd.Lights          = _lights;
    hsd.AmbientLight    = _ambientLight;
    hsd.Camera          = GC::DeviceAlloc<Camera>( _camera );
    hsd.Octree          = _octree;
    hsd.Image           = GC::DeviceAlloc<Image>( *( _image.get() ) );
    hsd.ViewPlane       = GC::DeviceAlloc<ViewPlane>( _viewPlane );
    hsd.BackgroundColor = GC::DeviceAlloc<Color>( _backgroundColor );


    // copy our image's contents over to the device
    _image->CopyHostToDevice();


    // create the device scene data (and copy from the host)
    DeviceSceneData* dsd = GC::DeviceAlloc<DeviceSceneData>( hsd );
    if ( dsd == nullptr )
    {
        return;
    }


    // prepare for the kernel
    int32 imgWidth  = GetNextPowerOfTwo( _image->GetWidth() );
    int32 imgHeight = GetNextPowerOfTwo( _image->GetHeight() );
    dim3  blocks    = dim3( 4, 4 );
    dim3  grid      = dim3( imgHeight / blocks.x + ( ( imgHeight % blocks.x ) == 0 ? 0 : 1 ),
                            imgWidth  / blocks.y + ( ( imgWidth  % blocks.y ) == 0 ? 0 : 1 ) );

    // start a timer
    Timer timer;
    timer.Start();

    // run the kernel
    SceneRenderKernel<<<grid, blocks>>>( dsd );

    // check for errors
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "  Render kernel failed. Reason: ", cudaGetErrorString( err ) );
        return;
    }

    // wait for the kernel to finish executing
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "  Failed to synchronize device. Reason: ", cudaGetErrorString( err ) );
        return;
    }


    timer.Stop();


    // copy our image's contents back to the host
    _image->CopyDeviceToHost();



    Logger::Log( "  Done rendering." );
    Logger::Log( "  Kernel time: ", timer.GetElapsed(), " seconds" );
}

REX_NS_END