#ifndef __REX_PLANE_HXX
#define __REX_PLANE_HXX

#include "../Config.hxx"
#include "Geometry.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a plane as a geometric object.
/// </summary>
class Plane : public Geometry
{
    Vector3 _point;
    Vector3 _normal;

public:
    /// <summary>
    /// Creates a new plane.
    /// </summary>
    Plane();

    /// <summary>
    /// Creates a new plane.
    /// </summary>
    /// <param name="point">The point through which the plane passes.</param>
    /// <param name="normal">The normal representing the plane.</param>
    Plane( const Vector3& point, const Vector3& normal );

    /// <summary>
    /// Destroys this plane.
    /// </summary>
    ~Plane();

    /// <summary>
    /// Gets this plane's bounds.
    /// </summary>
    virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Gets the normal that defines this plane.
    /// </summary>
    const Vector3& GetNormal() const;

    /// <summary>
    /// Gets a point this plane passes through.
    /// </summary>
    const Vector3& GetPoint() const;

    /// <summary>
    /// Gets the geometry type of this sphere.
    /// </summary>
    virtual GeometryType GetType() const;

    /// <summary>
    /// Checks to see if the given ray hits this plane. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const;

    /// <summary>
    /// Performs the same thing as a normal ray hit, but for shadow rays.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    virtual bool ShadowHit( const Ray& ray, real64& tmin ) const;
};

REX_NS_END

#endif