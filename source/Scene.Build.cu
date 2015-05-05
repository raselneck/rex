#include <rex/Rex.hxx>
#include <GLFW/glfw3.h>
#include <stdio.h>

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
    clock_t startTime = clock();



    // create the lists and the ambient light
    data->Lights       = new DeviceList<Light*>();
    data->Geometry     = new DeviceList<Geometry*>();
    data->AmbientLight = new AmbientLight( Color::White(), 1.0f );



    // add a directional light
    DirectionalLight* dl = new DirectionalLight();
    dl->SetDirection( vec3( 1.0f, 1.0f, 1.0f ) );
    dl->SetRadianceScale( real32( 1.5f ) );
    data->Lights->Add( dl );



    // prepare some materials
    const real32 ka    = 0.25f;
    const real32 kd    = 0.75f;
    const real32 ks    = 0.30f;
    const real32 kpow  = 2.00f;
    const PhongMaterial white ( Color::White(),  ka, kd, ks, kpow );
    const PhongMaterial red   ( Color::Red(),    ka, kd, ks, kpow );
    const PhongMaterial green ( Color::Green(),  ka, kd, ks, kpow );
    const PhongMaterial blue  ( Color::Blue(),   ka, kd, ks, kpow );
    const PhongMaterial orange( Color::Orange(), ka, kd, ks, kpow );
    const PhongMaterial purple( Color::Purple(), ka, kd, ks, kpow );

    // add some spheres
    data->Geometry->Add( new Sphere( purple, vec3(   0.0,   0.0,   0.0 ), 10.0 ) );
    data->Geometry->Add( new Sphere( green,  vec3(  10.0,  10.0,  10.0 ),  6.0 ) );
    data->Geometry->Add( new Sphere( white,  vec3( -15.0, -15.0, -15.0 ), 12.0 ) );

    // add some triangles
    data->Geometry->Add( new Triangle( orange, vec3(), vec3(  20.0, 0.0, 0.0 ), vec3(  20.0,  20.0,  15.0 ) ) );
    data->Geometry->Add( new Triangle( blue,   vec3(), vec3( -20.0, 0.0, 0.0 ), vec3( -20.0, -20.0, -15.0 ) ) );




    // calculate the min and max of the bounds
    vec3 min, max;
    for ( uint32 i = 0; i < data->Geometry->GetSize(); ++i )
    {
        Geometry* geom = data->Geometry->Get( i );
        min = glm::min( min, geom->GetBounds().GetMin() );
        max = glm::max( max, geom->GetBounds().GetMax() );
    }

    // create the octree
    data->Octree = new Octree( min, max );

    // add the objects to the octree
    for ( uint32 i = 0; i < data->Geometry->GetSize(); ++i )
    {
        Geometry*   geom   = data->Geometry->Get( i );
        BoundingBox bounds = geom->GetBounds();
        data->Octree->Add( geom, bounds );
    }



    clock_t endTime = clock();
    clock_t elapsed = abs( endTime - startTime );
    printf( "Elapsed: %f\n", elapsed / 1E-9f );
}

// build the scene
bool Scene::Build( uint16 width, uint16 height )
{
    return Build( width, height, 1, false );
}

// build the scene
bool Scene::Build( uint16 width, uint16 height, int32 samples )
{
    return Build( width, height, samples, false );
}

// build the scene
bool Scene::Build( uint16 width, uint16 height, int32 samples, bool fullscreen )
{
    // if we're rendering to an image, create the image
    if ( _renderMode == SceneRenderMode::ToImage )
    {
        _image = new Image( width, height );
    }
    // if we're rendering to OpenGL...
    else if ( _renderMode == SceneRenderMode::ToOpenGL )
    {
        // create the OpenGL window
        GLWindowHints hints;
        hints.Resizable  = false;
        hints.Visible    = false;
        hints.Fullscreen = fullscreen;
        _window = new GLWindow( width, height, "REX", hints );


        // ensure it was created
        if ( !_window->WasCreated() )
        {
            REX_DEBUG_LOG( "Failed to create OpenGL window." );
            return false;
        }


        // set the window pointer to be this scene
        GLFWwindow* win = reinterpret_cast<GLFWwindow*>( _window->_handle );
        glfwSetWindowUserPointer( win, this );

        // set the window's callbacks
        glfwSetKeyCallback( win, Scene::OnKeyPress );

        // set the window's mouse input mode
        glfwSetInputMode( win, GLFW_CURSOR, GLFW_CURSOR_DISABLED );

        
        // now create the texture
        _texture = new GLTexture2D( _window->GetContext(), width, height );
    }
    else
    {
        REX_DEBUG_LOG( "Invalid scene render mode." );
        return false;
    }



    // set the background color
    _backgroundColor = Color( real32( 0.392157 ),
                              real32( 0.584314 ),
                              real32( 0.929412 ) ); // cornflower blue ;)
    if ( _renderMode == SceneRenderMode::ToOpenGL )
    {
        glClearColor( _backgroundColor.R, _backgroundColor.G, _backgroundColor.B, 1.0f );
    }



    // setup the view plane
    _viewPlane.Width       = width;
    _viewPlane.Height      = height;
    _viewPlane.SampleCount = samples;


    
    // prepare for calling the kernel
    SceneBuildData  sdHost   = { nullptr, nullptr, nullptr };
    SceneBuildData* sdDevice = nullptr;
    if ( cudaSuccess != cudaMalloc( (void**)( &sdDevice ), sizeof( SceneBuildData ) ) )
    {
        REX_DEBUG_LOG( "Failed to allocate space for scene data." );
        return false;
    }
    if ( cudaSuccess != cudaMemcpy( sdDevice, &sdHost, sizeof( SceneBuildData ), cudaMemcpyHostToDevice ) )
    {
        REX_DEBUG_LOG( "Failed to initialize device scene data." );
        return false;
    }

    // start a timer to get the actual build time
    Timer timer;
    timer.Start();

    // call the kernel
    SceneBuildKernel<<<1, 1 >>>( sdDevice );

    // check for errors
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Build failed. Reason: ", cudaGetErrorString( err ) );
        return false;
    }

    // wait for the kernel to finish executing
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to synchronize device. Reason: ", cudaGetErrorString( err ) );
        return false;
    }

    timer.Stop();

    // copy our data back
    if ( cudaSuccess != cudaMemcpy( &sdHost, sdDevice, sizeof( SceneBuildData ), cudaMemcpyDeviceToHost ) )
    {
        REX_DEBUG_LOG( "Failed to copy data from device." );
        return false;
    }



    // set our references
    _lights       = sdHost.Lights;
    _ambientLight = sdHost.AmbientLight;
    _geometry     = sdHost.Geometry;
    _octree       = sdHost.Octree;



    // configure the camera
    vec3 camPosition = vec3( 0.0f, 0.0f, 100.0f );
    vec3 camTarget   = vec3( 0.0f, 0.0f, 0.0f );
    _camera.LookAt( camPosition, camTarget );
    _camera.SetViewPlaneDistance( 2000.0f );
    _camera.Update();



    REX_DEBUG_LOG( "Build time: ", timer.GetElapsed(), " seconds" );
    return true;
}

REX_NS_END