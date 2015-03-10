#ifndef __REX_TRIANGLE_HXX
#define __REX_TRIANGLE_HXX

#include "Geometry.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a piece of triangular geometry.
/// </summary>
class Triangle : public Geometry
{
public:
    /// <summary>
    /// The triangle's first point.
    /// </summary>
    Vector3 P1;

    /// <summary>
    /// The triangle's second point.
    /// </summary>
    Vector3 P2;

    /// <summary>
    /// The triangle's third point.
    /// </summary>
    Vector3 P3;

    /// <summary>
    /// Creates a new triangle.
    /// </summary>
    Triangle();

    /// <summary>
    /// Destroys this triangle.
    /// </summary>
    virtual ~Triangle();

    /// <summary>
    /// Gets this triangle's bounds.
    /// </summary>
    virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Gets this triangle's normal.
    /// </summary>
    Vector3 GetNormal() const;

    /// <summary>
    /// Gets the geometry type of this triangle.
    /// </summary>
    virtual GeometryType GetType() const;

    /// <summary>
    /// Checks to see if the given ray hits this triangle. If it does, the shading
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