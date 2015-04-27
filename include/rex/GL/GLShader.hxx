#pragma once

#include "../Config.hxx"
#include "../OpenGL.hxx"

/// <summary>
/// A macro for defining inline GLSL code.
/// </summary>
/// <param name="version">The version of the GLSL code.</param>
/// <param name="source">The GLSL source code.</param>
#define GLSL(version, source) "#version " #version "\n" #source

REX_NS_BEGIN

/// <summary>
/// An enumeration of possible shader types.
/// </summary>
enum class GLShaderType : GLenum
{
    Vertex   = GL_VERTEX_SHADER,
    Fragment = GL_FRAGMENT_SHADER
};

/// <summary>
/// Defines an OpenGL shader.
/// </summary>
class GLShader
{
    const GLShaderType _type;
    Handle<GLuint>     _handle;

public:
    /// <summary>
    /// Creates a new OpenGL shader.
    /// </summary>
    /// <param name="type">The shader's type.</param>
    GLShader( GLShaderType type );

    /// <summary>
    /// Destroys this OpenGL shader.
    /// </summary>
    ~GLShader();

    /// <summary>
    /// Gets this shader's handle.
    /// </summary>
    GLuint GetHandle() const;

    /// <summary>
    /// Gets this shader's information log.
    /// </summary>
    String GetLog() const;

    /// <summary>
    /// Attempts to load the given file as a shader.
    /// </summary>
    /// <param name="fname">The file name.</param>
    bool LoadFile( const String& fname );

    /// <summary>
    /// Attempts to load the given source code as a shader.
    /// </summary>
    /// <param name="source">The source.</param>
    bool LoadSource( const String& source );
};

REX_NS_END