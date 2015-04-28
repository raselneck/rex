#define _USE_MATH_DEFINES
#include <rex/Rex.hxx>
#include <math.h>
#include <stdio.h>
#include "DeviceScene.hxx"

REX_NS_BEGIN

// the scene data
static DeviceSceneData* SceneData = nullptr;


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

// handles pre-rendering
bool Scene::OnPreRender()
{
    // make sure the camera is up to date
    _camera.Update();

    // check if we need to create the scene data
    if ( !SceneData )
    {
        // create the host scene data
        DeviceSceneData hsd =
        {
            _lights,
            _ambientLight,
            _octree,
            _camera,
            _viewPlane,
            _backgroundColor,
            nullptr
        };

        // set the pixel information
        if ( _renderMode == SceneRenderMode::ToImage )
        {
            _image->CopyHostToDevice();
            hsd.Pixels = _image->GetDeviceMemory();
        }
        else if ( _renderMode == SceneRenderMode::ToOpenGL )
        {
            hsd.Pixels = _texture->GetDeviceMemory();
        }
        else
        {
            REX_DEBUG_LOG( "Invalid render mode." );
            return false;
        }


        // create the device scene data (and copy from the host)
        SceneData = GC::DeviceAlloc<DeviceSceneData>( hsd );
        if ( SceneData == nullptr )
        {
            return false;
        }
    }

    // copy over the camera if we're rendering to OpenGL
    if ( _renderMode == SceneRenderMode::ToOpenGL )
    {
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy( (void*)( &( SceneData->Camera ) ),
                          &_camera,
                          sizeof( Camera ),
                          cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            REX_DEBUG_LOG( "Failed to copy camera. Reason: ", cudaGetErrorString( err ) );
            return false;
        }
    }

    return true;
}

// handle post-rendering
bool Scene::OnPostRender()
{
    // check for errors
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Render kernel failed. Reason: ", cudaGetErrorString( err ) );
        return false;
    }

    // wait for the kernel to finish executing
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to synchronize device. Reason: ", cudaGetErrorString( err ) );
        return false;
    }

    return true;
}

// renders the scene
void Scene::Render()
{
    // prepare for the kernel
    int32 imgWidth  = GetNextPowerOfTwo( _viewPlane.Width );
    int32 imgHeight = GetNextPowerOfTwo( _viewPlane.Width );
    dim3  blocks    = dim3( 16, 16 );
    dim3  grid      = dim3( imgHeight / blocks.x + ( ( imgHeight % blocks.x ) == 0 ? 0 : 1 ),
                            imgWidth  / blocks.y + ( ( imgWidth  % blocks.y ) == 0 ? 0 : 1 ) );


    // if we're rendering to the image...
    if ( _renderMode == SceneRenderMode::ToImage )
    {
        // ensure our pre-render preparation is good
        if ( !OnPreRender() )
        {
            return;
        }

        // we should time the render
        Timer timer;

        // run the kernel and time it
        timer.Start();
        LaunchRenderKernel( blocks, grid, SceneData );
        timer.Stop();

        // ensure post-rendering cleanup is good
        if ( !OnPostRender() )
        {
            return;
        }

        // copy the information back to the image
        _image->CopyDeviceToHost();

        // log the render time
        REX_DEBUG_LOG( "Rendering took ", timer.GetElapsed(), " seconds (~", 1 / timer.GetElapsed(), " FPS)" );
    }
    // and if we're rendering to OpenGL...
    else if ( _renderMode == SceneRenderMode::ToOpenGL )
    {
        // let's create a timer so we can measure FPS
        Timer  timer;
        real64 elapsed    = 0.0;
        real64 total      = 0.0;
        real64 tickCount  = 0.0;
        uint64 frameCount = 0;


        // create the texture renderer
        TextureRenderer renderer = TextureRenderer( _texture );


        // now let's show the window and start the loop
        _window->Show();
        while ( _window->IsOpen() )
        {
            timer.Start();

            // update the camera
            UpdateCamera( elapsed );

            // ensure our pre-render preparation is good
            if ( !OnPreRender() )
            {
                _window->Close();
                continue;
            }

            // call the scene render kernel
            LaunchRenderKernel( blocks, grid, SceneData );

            // ensure nothing went wrong
            if ( !OnPostRender() )
            {
                _window->Close();
                continue;
            }



            // update the OpenGL texture
            _texture->UpdateOpenGLTexture();

            // render the texture
            renderer.Render();



            // swap buffers and poll the window events
            _window->SwapBuffers();
            _window->PollEvents();

            // check on the FPS
            timer.Stop();
            elapsed    = timer.GetElapsed();
            total     += elapsed;
            tickCount += elapsed;
            ++frameCount;
            if ( tickCount >= 1.0 )
            {
                tickCount -= 1.0;
                REX_DEBUG_LOG( frameCount, " FPS" );
                frameCount = 0;
            }
        }
    }
}

REX_NS_END