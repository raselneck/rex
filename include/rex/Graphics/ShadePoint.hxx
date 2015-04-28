#pragma once

#include "../Config.hxx"
#include "../Math/Ray.hxx"
#include "../Math/Math.hxx"
#include "Color.hxx"

REX_NS_BEGIN

class Scene;
class Material;

/// <summary>
/// Defines shading point information.
/// </summary>
struct ShadePoint
{
    Ray                  Ray;
    vec3              HitPoint;
    vec3              Normal;
    real32               T;
    const rex::Material* Material;

    /// <summary>
    /// Creates a new shade point.
    /// </summary>
    __both__ ShadePoint();

    /// <summary>
    /// Destroys this shade point.
    /// </summary>
    __both__ ~ShadePoint();
};

REX_NS_END