#pragma once

#include "../Config.hxx"
#include "../OpenGL.hxx"

REX_NS_BEGIN

/// <summary>
/// An enumeration of possible shader types.
/// </summary>
enum class ShaderType
{
    Vertex   = GL_VERTEX_SHADER,
    Fragment = GL_FRAGMENT_SHADER
};

/// <summary>
/// Defines a shader.
/// </summary>
class Shader
{
    Handle<GLuint> _handle;
    ShaderType _type;
};

REX_NS_END