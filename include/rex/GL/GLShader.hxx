#pragma once

#include "../Config.hxx"
#include "../OpenGL.hxx"

REX_NS_BEGIN

/// <summary>
/// An enumeration of possible shader types.
/// </summary>
enum class GLShaderType
{
    Vertex   = GL_VERTEX_SHADER,
    Fragment = GL_FRAGMENT_SHADER
};

/// <summary>
/// Defines an OpenGL shader.
/// </summary>
class GLShader
{
    Handle<GLuint> _handle;
    GLShaderType _type;
};

REX_NS_END