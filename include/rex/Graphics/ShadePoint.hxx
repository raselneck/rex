#pragma once

#include "../Config.hxx"
#include "../Math/Ray.hxx"
#include "../Math/Math.hxx"
#include "Lights/AmbientLight.hxx"
#include "Geometry/Octree.hxx"
#include "Color.hxx"

REX_NS_BEGIN

class Scene;
class Material;

/// <summary>
/// Defines shading point information.
/// </summary>
struct ShadePoint
{
    Ray                 Ray;
    vec3                HitPoint;
    vec3                Normal;
    real32              T;
    const Material*     Material;
    const AmbientLight* AmbientLight;
    const Octree*       Octree;
    const Light* const* Lights;
    uint32              LightCount;

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