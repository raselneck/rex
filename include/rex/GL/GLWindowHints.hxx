#pragma once

#include "../Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a set of hints to use when creating a window.
/// </summary>
struct GLWindowHints
{
    /// <summary>
    /// Whether or not the window should be resizable.
    /// </summary>
    bool Resizable;

    /// <summary>
    /// Whether or not the window should be initially visible.
    /// </summary>
    bool Visible;

    /// <summary>
    /// Whether or not the window should be fullscreen.
    /// </summary>
    bool Fullscreen;

    /// <summary>
    /// Whether or not the window should synchronize with the monitor's vertical retrace.
    /// </summary>
    bool VSync;

    /// <summary>
    /// Creates new window hints.
    /// </summary>
    GLWindowHints();

    /// <summary>
    /// Destroys these window hints.
    /// </summary>
    ~GLWindowHints();
};

REX_NS_END