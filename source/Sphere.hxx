#ifndef __REX_SPHERE_HXX
#define __REX_SPHERE_HXX

#include "Config.hxx"
#include "Geometry.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a sphere as a geometric object.
/// </summary>
class Sphere : public Geometry
{
    Vector3 _center;
    real64  _radius;
    real64  _invRadius;

public:
    /// <summary>
    /// Creates a new sphere.
    /// </summary>
    Sphere();
    
    /// <summary>
    /// Creates a new sphere.
    /// </summary>
    /// <param name="center">The center of the sphere.</param>
    /// <param name="radius">The radius of the sphere.</param>
    Sphere( const Vector3& center, real64 radius );

    /// <summary>
    /// Destroys this sphere.
    /// </summary>
    ~Sphere();

    /// <summary>
    /// Gets this sphere's bounds.
    /// </summary>
    virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Gets this sphere's center.
    /// </summary>
    const Vector3& GetCenter() const;

    /// <summary>
    /// Gets this sphere's radius.
    /// </summary>
    real64 GetRadius() const;

    /// <summary>
    /// Gets the geometry type of this sphere.
    /// </summary>
    virtual GeometryType GetType() const;

    /// <summary>
    /// Checks to see if the given ray hits this sphere. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const;
};

REX_NS_END

#endif