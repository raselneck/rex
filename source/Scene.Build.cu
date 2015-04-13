#include <rex/Rex.hxx>

REX_NS_BEGIN

/// <summary>
/// Defines a set of scene build data.
/// </summary>
struct SceneBuildData
{
    DeviceList<Light*>*    Lights;
    AmbientLight*          AmbientLight;
    DeviceList<Geometry*>* Geometry;
    Octree*                Octree;
};

/// <summary>
/// The scene build kernel.
/// </summary>
__global__ void SceneBuildKernel( SceneBuildData* data )
{
    // create the lists and the ambient light
    data->Lights       = new DeviceList<Light*>();
    data->Geometry     = new DeviceList<Geometry*>();
    data->AmbientLight = new AmbientLight();



    // add a directional light
    DirectionalLight* dl = new DirectionalLight();
    dl->SetDirection( Vector3( 1, 1, 2 ) );
    dl->SetRadianceScale( real_t( 1.5 ) );
    data->Lights->Add( dl );



    // prepare a material
    const real_t ka    = 0.25f;
    const real_t kd    = 0.75f;
    const real_t ks    = 0.30f;
    const real_t kpow  = 2.00f;
    const PhongMaterial white( Color::White(), ka, kd, ks, kpow );

    // add a sphere
    Sphere* sp1 = new Sphere( white, Vector3(  0.0,  0.0,  0.0 ), 10.0 );
    Sphere* sp2 = new Sphere( white, Vector3( 10.0, 10.0, 10.0 ),  4.0 );
    data->Geometry->Add( sp1 );
    data->Geometry->Add( sp2 );

    



    // calculate the min and max of the bounds
    Vector3 min, max;
    for ( uint32 i = 0; i < data->Geometry->GetSize(); ++i )
    {
        Geometry* geom = data->Geometry->operator[]( i );
        min = Vector3::Min( min, geom->GetBounds().GetMin() );
        max = Vector3::Max( max, geom->GetBounds().GetMax() );
    }

    // create the octree
    data->Octree = new Octree( min, max );

    // add the objects to the octree
    for ( uint32 i = 0; i < data->Geometry->GetSize(); ++i )
    {
        Geometry* geom = data->Geometry->operator[]( i );
        data->Octree->Add( geom );
    }
}

// build the scene
bool Scene::Build( uint16 width, uint16 height )
{
    Logger::Log( "Building scene..." );

    // make sure the image isn't too large
    if ( width > 1024 || height > 1024 )
    {
        Logger::Log( "  Image is too large. Max dimensions are 1024x1024, given ", width, "x", height, "." );
        return false;
    }

    // create the image
    _image.reset( new Image( width, height ) );

    // set the background color
    _backgroundColor = Color( real_t( 0.392157 ),
                              real_t( 0.584314 ),
                              real_t( 0.929412 ) ); // cornflower blue ;)

    // setup the view plane
    _viewPlane.Width        = width;
    _viewPlane.Height       = height;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f / _viewPlane.Gamma;
    _viewPlane.SampleCount  = 9;


    
    // prepare for calling the kernel
    Logger::Log( "  Preparing for build kernel..." );
    SceneBuildData  sdHost   = { nullptr, nullptr, nullptr };
    SceneBuildData* sdDevice = nullptr;
    if ( cudaSuccess != cudaMalloc( (void**)( &sdDevice ), sizeof( SceneBuildData ) ) )
    {
        Logger::Log( "  Failed to allocate space for scene data." );
        return false;
    }
    if ( cudaSuccess != cudaMemcpy( sdDevice, &sdHost, sizeof( SceneBuildData ), cudaMemcpyHostToDevice ) )
    {
        Logger::Log( "  Failed to initialize device scene data." );
        return false;
    }

    // call the kernel
    SceneBuildKernel<<<1, 1 >>>( sdDevice );

    // check for errors
    if ( cudaSuccess != cudaGetLastError() )
    {
        Logger::Log( "  Scene build failed. Reason: ", cudaGetErrorString( cudaGetLastError() ) );
        return false;
    }

    // wait for the kernel to finish executing
    if ( cudaSuccess != cudaDeviceSynchronize() )
    {
        Logger::Log( "  Failed to synchronize device. Reason: ", cudaGetErrorString( cudaDeviceSynchronize() ) );
        return false;
    }

    // copy our data back
    if ( cudaSuccess != cudaMemcpy( &sdHost, sdDevice, sizeof( SceneBuildData ), cudaMemcpyDeviceToHost ) )
    {
        Logger::Log( "  Failed to copy data from device." );
        return false;
    }

    // set our references
    _lights       = sdHost.Lights;
    _ambientLight = sdHost.AmbientLight;
    _geometry     = sdHost.Geometry;
    _octree       = sdHost.Octree;





    // configure the camera
    _camera.SetPosition( 0.0, 0.0, 100.0 );
    _camera.SetTarget( 0.0, 0.0, 0.0 );
    _camera.SetUp( 0.0, 1.0, 0.0 );
    _camera.SetViewPlaneDistance( 2000.0 );
    _camera.CalculateOrthonormalVectors();

    return true;
}

REX_NS_END