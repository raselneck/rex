#pragma once

#include "../Config.hxx"
#include "Ray.hxx"
#include "Vector3.hxx"

REX_NS_BEGIN

/// <summary>
/// An enumeration of possible containment types. (Inspired from XNA.)
/// </summary>
enum class ContainmentType : int32
{
    Disjoint,
    Contains,
    Intersects
};

/// <summary>
/// Defines a bounding box.
/// </summary>
class BoundingBox
{
    Vector3 _min;
    Vector3 _max;

public:
    /// <summary>
    /// Creates a new bounding box.
    /// </summary>
    /// <param name="min">The "minimum" corner.</param>
    /// <param name="max">The "maximum" corner.</param>
    __both__ BoundingBox( const Vector3& min, const Vector3& max );

    /// <summary>
    /// Creates a new bounding box.
    /// </summary>
    /// <param name="minX">The "minimum" corner's X.</param>
    /// <param name="minY">The "minimum" corner's Y.</param>
    /// <param name="minZ">The "minimum" corner's Z.</param>
    /// <param name="maxX">The "maximum" corner's X.</param>
    /// <param name="maxY">The "maximum" corner's Y.</param>
    /// <param name="maxZ">The "maximum" corner's Z.</param>
    __both__ BoundingBox( real_t minX, real_t minY, real_t minZ,
                          real_t maxX, real_t maxY, real_t maxZ );

    /// <summary>
    /// Destroys this bounding box.
    /// </summary>
    __both__ ~BoundingBox();

    /// <summary>
    /// Checks to see if this bounding box contains the given bounding box.
    /// </summary>
    /// <param name="bbox">The bounding box.</param>
    __both__ ContainmentType Contains( const BoundingBox& bbox ) const;

    /// <summary>
    /// Gets the center of the bounding box.
    /// </summary>
    __both__ Vector3 GetCenter() const;

    /// <summary>
    /// Gets this bounding box's "minimum" corner.
    /// </summary>
    __both__ const Vector3& GetMin() const;

    /// <summary>
    /// Gets this bounding box's "maximum" corner.
    /// </summary>
    __both__ const Vector3& GetMax() const;

    /// <summary>
    /// Gets the size of the bounding box.
    /// </summary>
    __both__ Vector3 GetSize() const;

    /// <summary>
    /// Checks to see if this bounding box intersects the given ray.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to the collision.</param>
    __both__ bool Intersects( const Ray& ray, real_t& dist ) const;
};

REX_NS_END