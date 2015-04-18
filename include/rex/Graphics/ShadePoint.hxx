#pragma once

#include "../Config.hxx"
#include "../Math/Ray.hxx"
#include "../Math/Vector3.hxx"
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
    Vector3              HitPoint;
    Vector3              Normal;
    real_t               T;
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