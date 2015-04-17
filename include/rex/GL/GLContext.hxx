#pragma once

#include "../Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an OpenGL context;
/// </summary>
class GLContext
{
    mutable void* _handle;

    friend class GLWindow;

    /// <summary>
    /// Creates a new OpenGL context.
    /// </summary>
    /// <param name="handle">The context handle.</param>
    GLContext( void* handle );

public:
    /// <summary>
    /// Destroys this OpenGL context.
    /// </summary>
    ~GLContext();

    /// <summary>
    /// Checks to see if this context is the current one.
    /// </summary>
    bool IsCurrent() const;

    /// <summary>
    /// Makes this OpenGL context the active one on the current thread.
    /// </summary>
    void MakeCurrent() const;

    /// <summary>
    /// Implicitly converts this context to a boolean to determine whether or not it exists.
    /// </summary>
    operator bool() const;
};

REX_NS_END