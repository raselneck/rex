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
    Ray             Ray;
    Vector3         HitPoint;
    Vector3         LocalHitPoint;
    Vector3         Normal;
    Vector3         Direction; // same as ray direction??
    real64          T;
    rex::Scene*     Scene;
    rex::Material*  Material;
    int32           RecursDepth;
    bool            HasHit;

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