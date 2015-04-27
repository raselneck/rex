#pragma once

#include "GLShader.hxx"
#include "GLTexture2D.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an OpenGL shader program.
/// </summary>
class GLShaderProgram
{
    /// <summary>
    /// Defines shader program data.
    /// </summary>
    struct ProgramData
    {
        GLuint   GLHandle;
        GLShader VertexShader;
        GLShader FragmentShader;

        /// <summary>
        /// Creates new program data.
        /// </summary>
        ProgramData();

        /// <summary>
        /// Destroys this program data.
        /// </summary>
        ~ProgramData();
    };

    Handle<ProgramData> _handle;

public:
    /// <summary>
    /// Creates a new OpenGL shader program.
    /// </summary>
    GLShaderProgram();

    /// <summary>
    /// Destroys this OpenGL shader program.
    /// </summary>
    ~GLShaderProgram();

    /// <summary>
    /// Tells OpenGL to use this shader program.
    /// </summary>
    void Use() const;

    /// <summary>
    /// Attempts to link this program with its vertex and fragment shaders.
    /// </summary>
    bool Link() const;

    /// <summary>
    /// Gets this program's information log.
    /// </summary>
    String GetLog() const;

    /// <summary>
    /// Finds the location of the uniform with the given name.
    /// </summary>
    /// <param name="name">The uniform name.</param>
    int FindUniformLocation( const char* name ) const;

    /// <summary>
    /// Gets this program's vertex shader.
    /// </summary>
    GLShader& GetVertexShader();

    /// <summary>
    /// Gets this program's fragment shader.
    /// </summary>
    GLShader& GetFragmentShader();

    /// <summary>
    /// Sets a texture uniform in this program.
    /// </summary>
    /// <param name="index">The uniform index.</param>
    /// <param name="texture">The texture.</param>
    /// <param name="unit">The texture unit to bind to.</param>
    void SetTexture( int index, const GLTexture2D& texture, int unit );
};

REX_NS_END