#ifndef __REX_SHADEPOINT_HXX
#define __REX_SHADEPOINT_HXX

#include "../Config.hxx"
#include "../Utility/Color.hxx"
#include "../Utility/Ray.hxx"
#include "../Utility/Vector3.hxx"

REX_NS_BEGIN

class Scene;
class Material;

/// <summary>
/// Defines shading point information.
/// </summary>
struct ShadePoint
{
    rex::Ray            Ray;
    Vector3             HitPoint;
    Vector3             LocalHitPoint;
    Vector3             Normal;
    Vector3             Direction; // same as ray direction??
    real64              T;
    rex::Scene* const   Scene;
    rex::Material*      Material;
    int32               RecursDepth;
    bool                HasHit;

    /// <summary>
    /// Creates a new shade point.
    /// </summary>
    /// <param name="scene">The scene this shade point is in.</param>
    ShadePoint( rex::Scene* scene );

    /// <summary>
    /// Destroys this shade point.
    /// </summary>
    ~ShadePoint();

    /// <summary>
    /// Resets this shade point.
    /// </summary>
    void Reset();
};

REX_NS_END

#endif