#ifndef __REX_PLANE_HXX
#define __REX_PLANE_HXX

#include "Config.hxx"
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
    /// Creates a new plane.
    /// </summary>
    /// <param name="point">The point through which the plane passes.</param>
    /// <param name="normal">The normal representing the plane.</param>
    /// <param name="color">The color of the plane.</param>
    Plane( const Vector3& point, const Vector3& normal, const Color& color );

    /// <summary>
    /// Destroys this plane.
    /// </summary>
    ~Plane();

    /// <summary>
    /// Gets this plane's bounds.
    /// </summary>
    virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Gets a point this plane passes through.
    /// </summary>
    const Vector3& GetPoint() const;

    /// <summary>
    /// Gets the normal that defines this plane.
    /// </summary>
    const Vector3& GetNormal() const;

    /// <summary>
    /// Checks to see if the given ray hits this plane. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const;
};

REX_NS_END

#endif