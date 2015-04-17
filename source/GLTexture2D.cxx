#include <rex/GL/GLTexture2D.hxx>
#include <rex/Utility/Logger.hxx>

REX_NS_BEGIN

// create texture handle data
Handle<GLTexture2D::HandleData> GLTexture2D::CreateHandleData( GLContext& context, uint32 width, uint32 height )
{
    HandleData handle;

    // create the texture
    glGenTextures( 1, &( handle.GLHandle ) );

    // initialize the texture to be the given size
    glBindTexture( GL_TEXTURE_2D, handle.GLHandle );
    glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr );
    glBindTexture( GL_TEXTURE_2D, 0 );

    // create the CUDA graphics resource
    cudaError_t err = cudaGraphicsGLRegisterImage( &( handle.CudaHandle ),
                                                   handle.GLHandle,
                                                   GL_TEXTURE_2D,
                                                   cudaGraphicsRegisterFlagsNone );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to register GL texture. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }

    // map the CUDA graphics resource to a CUDA array
    err = cudaGraphicsSubResourceGetMappedArray( handle.CudaArray,
                                                 handle.CudaHandle,
                                                 0, 0 );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to map graphics resource to array. Reason: ", cudaGetErrorString( err ) );
        return nullptr;
    }

    // TODO : http://3dgep.com/opengl-interoperability-with-cuda/


    // return the handle
    return Handle<HandleData>( new HandleData( handle ), DeleteHandleData );
}

// destroy texture handle data
void GLTexture2D::DeleteHandleData( void* voidHandleData )
{
    HandleData* handleData = reinterpret_cast<HandleData*>( voidHandleData );

    // TODO : Need to delete the CUDA handle?

    // delete the OpenGL texture
    glDeleteTextures( 1, &( handleData->GLHandle ) );

    // set everything to 0
    handleData->CudaHandle = nullptr;
    handleData->GLHandle   = 0;

    // delete the handle data
    delete handleData;
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

REX_NS_END