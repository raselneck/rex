#include "DeviceScene.hxx"

REX_NS_BEGIN

/// <summary>
/// The scene render kernel.
/// </summary>
/// <param name="sd">The scene data.</param>
__global__ void SceneRenderKernel( DeviceSceneData* sd )
{
    // get the image coordinates
    const int32      x  = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    const int32      y  = ( blockIdx.y * blockDim.y ) + threadIdx.y;
    const ViewPlane& vp = sd->ViewPlane;

    if ( x >= vp.Width || y >= vp.Height )
    {
        return;
    }

    // prepare for the tracing!!
    const Octree*    octree     = sd->Octree;
    const real_t     invSamples = 1.0f / vp.SampleCount;
    const real_t     half       = real_t( 0.5 );
    const int32      n          = static_cast<int32>( sqrtf( vp.SampleCount ) );
    const real_t     invn       = 1.0 / n;
    Color            color      = Color::Black();
    real_t           t          = 0;
    int32            sy         = 0;
    int32            sx         = 0;
    Ray              ray        = Ray( sd->Camera.GetPosition(), Vector3( 0, 0, 1 ) );
    Vector2          samplePoint;
    ShadePoint       shadePoint;

    // sample
    for ( sy = 0; sy < n; ++sy )
    {
        for ( sx = 0; sx < n; ++sx )
        {
            // get the pixel point
            samplePoint.X = x - half * vp.Width  + ( sx + half ) * invn;
            samplePoint.Y = y - half * vp.Height + ( sy + half ) * invn;


            // set the ray direction
            ray.Direction = sd->Camera.GetRayDirection( samplePoint );


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
                color += sd->BackgroundColor;
            }
        }
    }


    // set the pixel!
    color *= invSamples;
    uint32 index = x + y * vp.Width;
    sd->Pixels[ index ] = color.ToUChar4();
}

REX_NS_END