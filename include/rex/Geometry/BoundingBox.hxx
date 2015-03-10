#ifndef __REX_BOUNDINGBOX_HXX
#define __REX_BOUNDINGBOX_HXX

#include "../Config.hxx"
#include "../Utility/Ray.hxx"
#include "../Utility/Vector3.hxx"

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
    BoundingBox( const Vector3& min, const Vector3& max );

    /// <summary>
    /// Creates a new bounding box.
    /// </summary>
    /// <param name="minX">The "minimum" corner's X.</param>
    /// <param name="minY">The "minimum" corner's Y.</param>
    /// <param name="minZ">The "minimum" corner's Z.</param>
    /// <param name="maxX">The "maximum" corner's X.</param>
    /// <param name="maxY">The "maximum" corner's Y.</param>
    /// <param name="maxZ">The "maximum" corner's Z.</param>
    BoundingBox( real64 minX, real64 minY, real64 minZ,
                 real64 maxX, real64 maxY, real64 maxZ );

    /// <summary>
    /// Destroys this bounding box.
    /// </summary>
    ~BoundingBox();

    /// <summary>
    /// Checks to see if this bounding box contains the given bounding box.
    /// </summary>
    /// <param name="bbox">The bounding box.</param>
    ContainmentType Contains( const BoundingBox& bbox ) const;

    /// <summary>
    /// Gets the center of the bounding box.
    /// </summary>
    Vector3 GetCenter() const;

    /// <summary>
    /// Gets this bounding box's "minimum" corner.
    /// </summary>
    const Vector3& GetMin() const;

    /// <summary>
    /// Gets this bounding box's "maximum" corner.
    /// </summary>
    const Vector3& GetMax() const;

    /// <summary>
    /// Gets the size of the bounding box.
    /// </summary>
    Vector3 GetSize() const;

    /// <summary>
    /// Checks to see if this bounding box intersects the given ray.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to the collision.</param>
    bool Intersects( const Ray& ray, real64& dist ) const;

    /// <summary>
    /// Sets this bounding box's "minimum" corner.
    /// </summary>
    /// <param name="min">The new "minimum" corner.</param>
    void SetMin( const Vector3& min );

    /// <summary>
    /// Sets this bounding box's "maximum" corner.
    /// </summary>
    /// <param name="max">The new "maximum" corner.</param>
    void SetMax( const Vector3& max );
};

REX_NS_END

#endif