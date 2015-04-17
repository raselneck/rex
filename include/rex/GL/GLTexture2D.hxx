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
    Handle<GLuint> _handle;

public:
    /// <summary>
    /// Creates a new 2D texture.
    /// </summary>
    /// <param name="context">The OpenGL context to use when creating this texture.</param>
    GLTexture2D( GLContext& context );

    /// <summary>
    /// Destroys this texture.
    /// </summary>
    ~GLTexture2D();
};

REX_NS_END