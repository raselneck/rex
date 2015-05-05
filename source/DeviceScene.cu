#include "DeviceScene.hxx"

REX_NS_BEGIN

// launches the scene render kernel
void LaunchRenderKernel( const dim3& blocks, const dim3& grid, DeviceSceneData* sceneData )
{
    SceneRenderKernel<<<grid, blocks>>>( sceneData );
}

// the scene render kernel, where the magic happens
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
    const Octree* octree     = sd->Octree;
    const real32  invSamples = 1.0f / vp.SampleCount;
    const int32   n          = static_cast<int32>( sqrtf( vp.SampleCount ) );
    const real32  invn       = 1.0f / n;
    Color         color      = Color::Black();
    real32        t          = 0.0f;
    int32         sy         = 0;
    int32         sx         = 0;
    Ray           ray        = Ray( sd->Camera.GetPosition(), vec3( 0, 0, 1 ) );
    vec2          samplePoint;
    ShadePoint    shadePoint;


    // configure the shade point
    shadePoint.Octree       = sd->Octree;
    shadePoint.AmbientLight = sd->AmbientLight;
    shadePoint.LightCount   = sd->Lights->GetSize();
    shadePoint.Lights       = &( sd->Lights->Get( 0 ) );


    // sample the scene!
    for ( sy = 0; sy < n; ++sy )
    {
        for ( sx = 0; sx < n; ++sx )
        {
            // get the pixel point
            samplePoint.x = x - ( 0.5f * vp.Width  ) + ( ( sx + 0.5f ) * invn );
            samplePoint.y = y - ( 0.5f * vp.Height ) + ( ( sy + 0.5f ) * invn );


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
                color += mat->Shade( shadePoint );
            }
            else
            {
                color += sd->BackgroundColor;
            }
        }
    }


    // set the pixel!
    color *= invSamples;
    sd->Pixels[ x + y * vp.Width ] = color.ToUChar4();
}

REX_NS_END