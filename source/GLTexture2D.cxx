#include <rex/GL/GLTexture2D.hxx>
#include <rex/Utility/Logger.hxx>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>

REX_NS_BEGIN

// create texture handle data
GLTexture2D::HandleData* GLTexture2D::CreateHandleData( GLContext& context, uint32 width, uint32 height )
{
    HandleData* handle = new HandleData();



    // create the OpenGL texture
    glGenTextures( 1, &( handle->GLHandle ) );

    // initialize the texture to be the given size
    glBindTexture( GL_TEXTURE_2D, handle->GLHandle );
    glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr );
    glBindTexture( GL_TEXTURE_2D, 0 );



    // create the CUDA graphics resource
    cudaError_t err = cudaGraphicsGLRegisterImage( &( handle->CudaHandle ), handle->GLHandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to register GL texture. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }



    // map the CUDA graphics resource to a CUDA array
    err = cudaGraphicsSubResourceGetMappedArray( &( handle->CudaArray ), handle->CudaHandle, 0, 0 );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to map graphics resource to array. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }



    // create the CUDA texture reference
    handle->CudaTextureRef.normalized = true;
    handle->CudaTextureRef.filterMode = cudaFilterModePoint;
    handle->CudaTextureRef.addressMode[ 0 ] = cudaAddressModeClamp;
    handle->CudaTextureRef.addressMode[ 1 ] = cudaAddressModeClamp;
    handle->CudaTextureRef.channelDesc.x = 8;
    handle->CudaTextureRef.channelDesc.y = 8;
    handle->CudaTextureRef.channelDesc.z = 8;
    handle->CudaTextureRef.channelDesc.w = 0;
    handle->CudaTextureRef.channelDesc.f = cudaChannelFormatKindUnsigned;

    // bind the CUDA array to a texture object
    cudaBindTextureToArray( &( handle->CudaTextureRef ), handle->CudaArray, &( handle->CudaTextureRef.channelDesc ) );


    
    // allocate the texture memory array
    uint_t memSize = width * height * sizeof( uchar3 );
    cudaMalloc( &( handle->TextureMemory ), memSize );



    // return the handle
    return handle;
}

// create texture
GLTexture2D::GLTexture2D( GLContext& context, uint32 width, uint32 height )
    : _handle( CreateHandleData( context, width, height ) ),
      _width( width ),
      _height( height )
{
    // ensure the desired context is the current one
    if ( !context.IsCurrent() )
    {
        context.MakeCurrent();
    }

    
}

// destroy texture
GLTexture2D::~GLTexture2D()
{
    // delete the texture memory
    cudaFree( _handle->TextureMemory );

    // delete the OpenGL texture
    glDeleteTextures( 1, &( _handle->GLHandle ) );


    // set everything to 0
    memset( _handle, 0, sizeof( HandleData ) );

    // delete the handle data
    delete _handle;
}

// get width
int32 GLTexture2D::GetWidth() const
{
    return _width;
}

// get height
int32 GLTexture2D::GetHeight() const
{
    return _height;
}

// get CUDA memory
uchar3* GLTexture2D::GetCudaMemory()
{
    return _handle->TextureMemory;
}

// update GL texture
void GLTexture2D::UpdateOpenGLTexture()
{
    uint_t bufferSize = _width * _height * sizeof( uchar3 );
    cudaError_t err = cudaMemcpyToArray( _handle->CudaArray, 0, 0, _handle->TextureMemory, bufferSize, cudaMemcpyDeviceToDevice );
}

REX_NS_END