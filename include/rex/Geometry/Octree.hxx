#ifndef __REX_OCTREE_HXX
#define __REX_OCTREE_HXX

#include "../Config.hxx"
#include "BoundingBox.hxx"
#include <array>
#include <utility> // std::pair
#include <vector>

REX_NS_BEGIN

class Geometry;

/// <summary>
/// Defines an octree meant for spatially partitioning static objects based on their bounding boxes.
/// </summary>
class Octree
{
    std::vector<std::pair<BoundingBox, const Geometry*>> _objects;
    std::array<Octree*, 8>  _children;
    BoundingBox             _bounds;

    /// <summary>
    /// Checks to see if this octree has subdivided.
    /// </summary>
    bool HasSubdivided() const;

    /// <summary>
    /// Subdivides this octree.
    /// </summary>
    void Subdivide();

    /// <summary>
    /// Queries this octree for the pieces of geometry that a given ray intersects.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="objects">The objects list to check.</param>
    /// <param name="clear">True to clear the list, false to leave it alone.</param>
    void QueryIntersections( const Ray& ray, std::vector<const Geometry*>& objects, bool clear ) const;

public:
    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="bounds">The octree's bounds.</param>
    Octree( const BoundingBox& bounds );

    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="min">The minimum corner of the bounds.</param>
    /// <param name="max">The maximum corner of the bounds.</param>
    Octree( const Vector3& min, const Vector3& max );

    /// <summary>
    /// Destroys this octree.
    /// </summary>
    ~Octree();

    /// <summary>
    /// Queries this octree for the pieces of geometry that a given ray intersects.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="objects">The objects list to check.</param>
    void QueryIntersections( const Ray& ray, std::vector<const Geometry*>& objects ) const;

    /// <summary>
    /// Adds the given bounding box to this octree.
    /// </summary>
    /// <param name="geometry">The piece of geometry to add.</param>
    bool Add( const Geometry* geometry );
};

REX_NS_END

#endif