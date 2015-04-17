#pragma once

#include "../Config.hxx"
#include "../OpenGL.hxx"
#include "GLContext.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an OpenGL 2D texture.
/// </summary>
class GLTexture2D
{
    /// <summary>
    /// Defines an OpenGL 2D texture.
    /// </summary>
    struct HandleData
    {
        GLuint GLHandle;
        cudaGraphicsResource* CudaHandle;
        cudaArray_t* CudaArray;
        textureReference CudaTexture;
    };

    Handle<HandleData> _handle;
    const int32 _width;
    const int32 _height;

    /// <summary>
    /// Creates 2D texture handle data.
    /// </summary>
    /// <param name="context">The OpenGL context to use when creating this texture.</param>
    /// <param name="width">The width of the texture.</param>
    /// <param name="height">The height of the texture.</param>
    static Handle<HandleData> CreateHandleData( GLContext& context, uint32 width, uint32 height );

    /// <summary>
    /// Destroys 2D texture handle data.
    /// </summary>
    /// <param name="handleData">The handle data.</param>
    static void DeleteHandleData( void* handleData );

public:
    /// <summary>
    /// Creates a new 2D texture.
    /// </summary>
    /// <param name="context">The OpenGL context to use when creating this texture.</param>
    /// <param name="width">The width of the texture.</param>
    /// <param name="height">The height of the texture.</param>
    GLTexture2D( GLContext& context, uint32 width, uint32 height );

    /// <summary>
    /// Destroys this texture.
    /// </summary>
    ~GLTexture2D();

    /// <summary>
    /// Gets this texture's width.
    /// </summary>
    int32 GetWidth() const;

    /// <summary>
    /// Gets this texture's height.
    /// </summary>
    int32 GetHeight() const;
};

REX_NS_END