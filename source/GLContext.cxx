#include <rex/GL/GLContext.hxx>
#include <rex/Utility/Logger.hxx>
#include <rex/OpenGL.hxx>
#include <GLFW/glfw3.h>

#pragma warning( disable : 4800 ) // * -> bool conversion

REX_NS_BEGIN

// create new context
GLContext::GLContext( void* handle )
    : _handle( handle )
{
}

// destroy context
GLContext::~GLContext()
{
    _handle = nullptr;
}

// make context current
void GLContext::MakeCurrent() const
{
    GLFWwindow* window = static_cast<GLFWwindow*>( _handle );
    glfwMakeContextCurrent( window );
}

// convert context to bool
GLContext::operator bool() const
{
    return static_cast<bool>( _handle );
}

REX_NS_END