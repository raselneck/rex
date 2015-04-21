#include <rex/GL/GLTexture2D.hxx>
#include <rex/Utility/Logger.hxx>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>

REX_NS_BEGIN

// create texture handle data
GLTexture2D::HandleData* GLTexture2D::CreateHandleData( GLContext& context, uint32 width, uint32 height )
{
    // ensure the desired context is the current one
    if ( !context.IsCurrent() )
    {
        context.MakeCurrent();
    }



    // create the handle
    HandleData* handle = new HandleData();



    // create the OpenGL texture
    glGenTextures( 1, &( handle->GLHandle ) );

    // initialize the texture to be the given size
    glBindTexture( GL_TEXTURE_2D, handle->GLHandle );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );
    //  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_POINT );
    //  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_POINT );
    //  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
    //  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
    glBindTexture( GL_TEXTURE_2D, 0 );



    // create the CUDA graphics resource
    cudaError_t err = cudaGraphicsGLRegisterImage( &( handle->CudaHandle ), handle->GLHandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to register GL texture. Reason: ", cudaGetErrorString( err ) );
        delete handle;
        return nullptr;
    }

    // map the CUDA graphics resource to a CUDA array
    err = cudaGraphicsMapResources( 1, &( handle->CudaHandle ), nullptr );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to map graphics resource to array. Reason: ", cudaGetErrorString( err ) );
        delete handle;
        return nullptr;
    }

    // get the mapped array
    err = cudaGraphicsSubResourceGetMappedArray( &( handle->CudaArray ), handle->CudaHandle, 0, 0 );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to get mapped array. Reason: ", cudaGetErrorString( err ) );
        delete handle;
        return nullptr;
    }


    // create the CUDA texture reference
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    texture<uchar4, 2, cudaReadModeElementType> tex;
    tex.addressMode[ 0 ] = cudaAddressModeClamp;
    tex.addressMode[ 1 ] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;


    // bind the CUDA array to a texture object
    err = cudaBindTextureToArray( tex, handle->CudaArray, channelDesc );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to bind texture to array. Reason: ", cudaGetErrorString( err ) );
        delete handle;
        return nullptr;
    }



    // allocate the texture memory array
    uint_t memSize = width * height * sizeof( uchar4 );
    cudaMalloc( &( handle->TextureMemory ), memSize );
    cudaMemset( handle->TextureMemory, 0, memSize );



    // return the handle
    return handle;
}

// create texture
GLTexture2D::GLTexture2D( GLContext& context, uint32 width, uint32 height )
    : _handle( CreateHandleData( context, width, height ) ),
    _width( width ),
    _height( height )
{
}

// destroy texture
GLTexture2D::~GLTexture2D()
{
    if ( _handle )
    {
        // delete the texture memory
        cudaFree( _handle->TextureMemory );

        // un-map the resources
        cudaGraphicsUnmapResources( 1, &( _handle->CudaHandle ) );

        // unregister the resource
        cudaGraphicsUnregisterResource( _handle->CudaHandle );

        // delete the OpenGL texture
        glDeleteTextures( 1, &( _handle->GLHandle ) );

        // set everything to 0
        memset( _handle, 0, sizeof( HandleData ) );

        // delete the handle data
        delete _handle;
    }
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