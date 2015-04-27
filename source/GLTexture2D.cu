#include <rex/GL/GLTexture2D.hxx>
#include <rex/Utility/Logger.hxx>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>


// TODO : http://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda


REX_NS_BEGIN

// create texture handle data
Handle<GLTexture2D::HandleData> GLTexture2D::CreateHandleData( GLContext& context, uint32 width, uint32 height )
{
    // ensure the desired context is the current one
    if ( !context.IsCurrent() )
    {
        context.MakeCurrent();
    }


    // create the handle
    Handle<HandleData> handle = Handle<HandleData>( new HandleData(), DestroyHandleData );
    handle->CudaArray     = nullptr;
    handle->CudaResource  = nullptr;
    handle->TextureMemory = nullptr;
    handle->GLHandle      = 0;


    // create the OpenGL texture
    glGenTextures( 1, &( handle->GLHandle ) );

    // initialize the texture to be the given size
    glBindTexture  ( GL_TEXTURE_2D, handle->GLHandle );
    glTexImage2D   ( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST   );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST   );
    glBindTexture  ( GL_TEXTURE_2D, 0 );


    // register the image with CUDA
    cudaError_t err = cudaSuccess;
    err = cudaGraphicsGLRegisterImage( &( handle->CudaResource ),
                                       handle->GLHandle,
                                       GL_TEXTURE_2D,
                                       cudaGraphicsRegisterFlagsNone );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to register OpenGL texture. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }


    // map the resources
    err = cudaGraphicsMapResources( 1,
                                    &( handle->CudaResource ),
                                    nullptr );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to map resources. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }


    // get the mapped array
    err = cudaGraphicsSubResourceGetMappedArray( &( handle->CudaArray ),
                                                 handle->CudaResource,
                                                 0,
                                                 0 );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to get mapped array. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }
    

    //// create the CUDA texture reference
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    //texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex = { 0 };
    //tex.addressMode[ 0 ] = cudaAddressModeClamp;
    //tex.addressMode[ 1 ] = cudaAddressModeClamp;
    //tex.filterMode       = cudaFilterModePoint;
    //tex.normalized       = true;
    //
    //
    //// bind the CUDA array to a texture object (THIS is where the error happens)
    //err = cudaBindTextureToArray( tex, handle->CudaArray );
    //if ( err != cudaSuccess )
    //{
    //    REX_DEBUG_LOG( "Failed to bind texture to array. Reason: ", cudaGetErrorString( err ) );
    //    delete handle;
    //    return nullptr;
    //}


    // allocate the texture memory
    uint_t size = width * height * sizeof( uchar4 );
    cudaMalloc( &( handle->TextureMemory ), size );
    cudaMemset( handle->TextureMemory, 0, size );


    // return the handle
    return handle;
}

// destroy texture handle data
void GLTexture2D::DestroyHandleData( void* data )
{
    HandleData* hd = reinterpret_cast<HandleData*>( data );

    // delete the texture memory
    if ( hd->TextureMemory )
    {
        cudaFree( hd->TextureMemory );
        hd->TextureMemory = nullptr;
    }

    // dispose of the graphics resource
    if ( hd->CudaResource )
    {
        // un-map the resources
        cudaGraphicsUnmapResources( 1, &( hd->CudaResource ), nullptr );

        // unregister the resource
        cudaGraphicsUnregisterResource( hd->CudaResource );

        hd->CudaResource = nullptr;
        hd->CudaArray    = nullptr;
    }

    // delete the OpenGL texture
    if ( hd->GLHandle )
    {
        glDeleteTextures( 1, &( hd->GLHandle ) );
        hd->GLHandle = 0;
    }

    // delete the handle data
    delete hd;
}

// create texture
GLTexture2D::GLTexture2D( GLContext& context, uint32 width, uint32 height )
    : _handle( CreateHandleData( context, width, height ) )
    , _width( width )
    , _height( height )
{
}

// destroy texture
GLTexture2D::~GLTexture2D()
{
}

// get width
uint32 GLTexture2D::GetWidth() const
{
    return _width;
}

// get height
uint32 GLTexture2D::GetHeight() const
{
    return _height;
}

// get OpenGL texture handle
GLuint GLTexture2D::GetOpenGLHandle() const
{
    return _handle->GLHandle;
}

// get CUDA memory
uchar4* GLTexture2D::GetDeviceMemory()
{
    return _handle->TextureMemory;
}

// update GL texture
void GLTexture2D::UpdateOpenGLTexture()
{
    uint_t bufferSize = _width * _height * sizeof( uchar4 );
    cudaError_t err = cudaMemcpyToArray( _handle->CudaArray, 0, 0, _handle->TextureMemory, bufferSize, cudaMemcpyDeviceToDevice );
}

REX_NS_END