#pragma once

#include "../GL/GLTexture2D.hxx"
#include "../GL/GLShaderProgram.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an OpenGL texture renderer.
/// </summary>
class TextureRenderer
{
    const GLTexture2D*  _texture;
    GLShaderProgram     _program;
    GLuint              _vbo;
    GLuint              _vao;
    GLint               _uniformTexLocation;

    /// <summary>
    /// Creates the OpenGL buffer objects.
    /// </summary>
    void CreateBufferObjects();

    /// <summary>
    /// Creates the OpenGL shader program.
    /// </summary>
    void CreateShaderProgram();

public:
    /// <summary>
    /// Creates a new OpenGL texture renderer.
    /// </summary>
    /// <param name="texture">The texture.</param>
    TextureRenderer( const GLTexture2D* texture );

    /// <summary>
    /// Destroys this OpenGL texture renderer.
    /// </summary>
    ~TextureRenderer();

    /// <summary>
    /// Renders the texture to the whole window.
    /// </summary>
    void Render();
};

REX_NS_END