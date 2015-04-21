#include "DeviceScene.hxx"

REX_NS_BEGIN

/// <summary>
/// The scene render kernel.
/// </summary>
/// <param name="sd">The scene data.</param>
__global__ void SceneRenderKernel( DeviceSceneData* sd )
{
    // get the image coordinates
    int32  x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    int32  y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
    Image* image = sd->Image;

    if ( x >= image->GetWidth() || y >= image->GetHeight() )
    {
        return;
    }

    // prepare for the tracing!!
    const Color&     bgColor = *sd->BackgroundColor;
    const Camera*    camera = sd->Camera;
    const ViewPlane* vp = sd->ViewPlane;
    const Octree*    octree = sd->Octree;
    const real_t     invSamples = 1.0f / vp->SampleCount;
    const real_t     half = real_t( 0.5 );
    const int32      n = static_cast<int32>( sqrtf( vp->SampleCount ) );
    const real_t     invn = 1.0 / n;
    Color            color = Color::Black();
    real_t           t = 0;
    int32            sy = 0;
    int32            sx = 0;
    Ray              ray = Ray( camera->GetPosition(), Vector3( 0, 0, 1 ) );
    Vector2          samplePoint;
    ShadePoint       shadePoint;

    // sample
    for ( sy = 0; sy < n; ++sy )
    {
        for ( sx = 0; sx < n; ++sx )
        {
            // get the pixel point
            samplePoint.X = x - half * vp->Width + ( sx + half ) * invn;
            samplePoint.Y = y - half * vp->Height + ( sy + half ) * invn;


            // set the ray direction
            ray.Direction = camera->GetRayDirection( samplePoint );


            // hit the objects in the scene
            const Geometry* geom = octree->QueryIntersections( ray, t, shadePoint );
            if ( geom )
            {
                shadePoint.Ray = ray;
                shadePoint.T = t;

                // add to the color if the ray hit
                const Material* mat = shadePoint.Material;
                color += mat->Shade( shadePoint, sd->Lights, sd->Octree );
            }
            else
            {
                color += bgColor;
            }
        }
    }


    // set the pixel!
    color *= invSamples;
    if ( sd->RenderMode == RenderMode::Image )
    {
        image->SetDevicePixel( x, y, color );
    }
    else
    {
        uint64 index = x + y * image->GetWidth();
    }
}

REX_NS_END