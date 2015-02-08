#ifndef __REX_SHADEPOINT_HXX
#define __REX_SHADEPOINT_HXX
#pragma once

#include "Config.hxx"
#include "Color.hxx"
#include "Vector3.hxx"

REX_NS_BEGIN

class Scene;

/// <summary>
/// Defines shading point information.
/// </summary>
struct ShadePoint
{
    Vector3      HitPoint;
    Vector3      Normal;
    Color        Color;
    Scene* const ScenePtr;
    bool         HasHit;

    /// <summary>
    /// Creates a new shade point.
    /// </summary>
    /// <param name="scene">The scene this shade point is in.</param>
    ShadePoint( Scene* scene );

    /// <summary>
    /// Destroys this shade point.
    /// </summary>
    ~ShadePoint();
};

REX_NS_END

#endif