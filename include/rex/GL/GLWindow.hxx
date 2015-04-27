#pragma once

#include "../Config.hxx"
#include "GLContext.hxx"
#include "GLWindowHints.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an OpenGL window.
/// </summary>
class GLWindow
{
    friend class Scene;

    static uint32 _windowCount;
    mutable void* _handle;

    /// <summary>
    /// Initializes GLFW.
    /// </summary>
    static bool InitializeGlfw();

public:
    /// <summary>
    /// Creates a new window.
    /// </summary>
    /// <param name="width">The initial width of the window.</param>
    /// <param name="height">The initial height of the window.</param>
    GLWindow( int32 width, int32 height );

    /// <summary>
    /// Creates a new window.
    /// </summary>
    /// <param name="width">The initial width of the window.</param>
    /// <param name="height">The initial height of the window.</param>
    /// <param name="title">The initial title of the window.</param>
    GLWindow( int32 width, int32 height, const String& title );

    /// <summary>
    /// Creates a new window.
    /// </summary>
    /// <param name="width">The initial width of the window.</param>
    /// <param name="height">The initial height of the window.</param>
    /// <param name="title">The initial title of the window.</param>
    /// <param name="hints">The hints to use when creating the window.</param>
    GLWindow( int32 width, int32 height, const String& title, const GLWindowHints& hints );

    // TODO : Add constructor taking in another window to share contexts?

    /// <summary>
    /// Copies another GLWindow.
    /// </summary>
    /// <param name=""></param>
    GLWindow( const GLWindow& other );

    /// <summary>
    /// Destroys this window.
    /// </summary>
    ~GLWindow();

    /// <summary>
    /// Checks to see if this window is open.
    /// </summary>
    bool IsOpen() const;

    /// <summary>
    /// Gets this window's OpenGL context.
    /// </summary>
    GLContext GetContext() const;

    /// <summary>
    /// Checks to see if this window was created.
    /// </summary>
    bool WasCreated() const;

    /// <summary>
    /// Closes this window.
    /// </summary>
    void Close();

    /// <summary>
    /// Hides this window.
    /// </summary>
    void Hide();

    /// <summary>
    /// Polls this window's events.
    /// </summary>
    void PollEvents();

    /// <summary>
    /// Shows this window.
    /// </summary>
    void Show();

    /// <summary>
    /// Swaps this window's front and back buffers
    /// </summary>
    void SwapBuffers();

    /// <summary>
    /// Implicitly converts this window to a boolean to determine whether or not it exists.
    /// </summary>
    operator bool() const;
};

REX_NS_END