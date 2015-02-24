#ifndef __REX_OCTREE_HXX
#define __REX_OCTREE_HXX

#include "../Config.hxx"
#include "BoundingBox.hxx"
#include <vector>

// TODO : Implement this

REX_NS_BEGIN

/// <summary>
/// Defines an octree meant for spatially partitioning static objects based on their bounding boxes.
/// </summary>
class Octree
{
    std::vector<BoundingBox> _objects;
    BoundingBox              _bounds;
    Handle<Octree>           _children;

    /// <summary>
    /// Checks to see if this octree has subdivided.
    /// </summary>
    bool HasSubdivided() const;

    /// <summary>
    /// Subdivides this octree.
    /// </summary>
    void Subdivide();

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
    /// Adds the given bounding box to this octree.
    /// </summary>
    /// <param name="box">The bounds to remove.</param>
    void Add( const BoundingBox& box );
};

REX_NS_END

#endif