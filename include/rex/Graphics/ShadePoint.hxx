#pragma once

#include "../Config.hxx"
#include "../Math/Ray.hxx"
#include "../Math/Vector3.hxx"
#include "Color.hxx"

// TODO : Do we need the scene pointer? Can we make the SceneData struct public?

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
    Vector3              LocalHitPoint;
    Vector3              Normal;
    Vector3              Direction; // TODO : Same as ray direction?
    real_t               T;
    const rex::Scene*    Scene;
    const rex::Material* Material;
    bool                 HasHit;

    /// <summary>
    /// Creates a new shade point.
    /// </summary>
    /// <param name="scene">The scene this shade point is in.</param>
    __both__ ShadePoint( rex::Scene* scene );

    /// <summary>
    /// Destroys this shade point.
    /// </summary>
    __both__ ~ShadePoint();
};

REX_NS_END