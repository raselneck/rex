#pragma once

#include "../../Math/BoundingBox.hxx"
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
    uint32                  _maxItemCount;

    /// <summary>
    /// Checks to see if this octree has subdivided.
    /// </summary>
    bool HasSubdivided() const;

    /// <summary>
    /// Subdivides this octree.
    /// </summary>
    void Subdivide();

    /// <summary>
    /// Recursively queries this octree for the pieces of geometry that a given ray intersects.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="objects">The objects list to check.</param>
    void QueryIntersectionsRecurs( const Ray& ray, std::vector<const Geometry*>& objects ) const;

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
    /// Creates a new octree.
    /// </summary>
    /// <param name="min">The minimum corner of the bounds.</param>
    /// <param name="max">The maximum corner of the bounds.</param>
    /// <param name="maxItemCount">The maximum number of items to allow per-node before that node subdivides.</param>
    Octree( const Vector3& min, const Vector3& max, uint32 maxItemCount );

    /// <summary>
    /// Destroys this octree.
    /// </summary>
    ~Octree();

    /// <summary>
    /// Gets this octree's bounds.
    /// </summary>
    const BoundingBox& GetBounds() const;

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