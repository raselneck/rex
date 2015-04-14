#pragma once

#include "../../Math/BoundingBox.hxx"
#include "../../CUDA/DeviceList.hxx"

REX_NS_BEGIN

class Geometry;
struct ShadePoint;

/// <summary>
/// Defines a pairing between a bounding box and geometry.
/// </summary>
struct BoundsGeometryPair
{
    const Geometry* Geometry;
    BoundingBox Bounds;

    /// <summary>
    /// Creates a new bounding box / geometry pairing.
    /// </summary>
    __device__ BoundsGeometryPair();
};

/// <summary>
/// Defines an octree meant for spatially partitioning static objects based on their bounding boxes.
/// </summary>
class Octree
{
    REX_NONCOPYABLE_CLASS( Octree )
    REX_IMPLEMENT_DEVICE_MEM_OPS()

    BoundingBox                    _bounds;
    const uint_t                   _countBeforeSubivide;
    Octree*                        _children[ 8 ];
    DeviceList<BoundsGeometryPair> _objects;

    /// <summary>
    /// Checks to see if this octree has subdivided.
    /// </summary>
    __device__ bool HasSubdivided() const;

    /// <summary>
    /// Subdivides this octree.
    /// </summary>
    __device__ void Subdivide();

    /// <summary>
    /// Queries this octree for the nearest piece of geometry that a given ray intersects for realzies this time.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to the piece of geometry.</param>
    /// <param name="sp">The shade point data.</param>
    __device__ const Geometry* QueryIntersectionsForReal( const Ray& ray, real_t& dist, ShadePoint& sp ) const;

public:
    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="bounds">The octree's bounds.</param>
    __device__ Octree( const BoundingBox& bounds );

    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="min">The minimum corner of the bounds.</param>
    /// <param name="max">The maximum corner of the bounds.</param>
    __device__ Octree( const Vector3& min, const Vector3& max );

    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="bounds">The octree's bounds.</param>
    /// <param name="maxItemCount">The maximum number of items to allow per-node before that node subdivides.</param>
    __device__ Octree( const BoundingBox& bounds, uint_t maxItemCount );

    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="min">The minimum corner of the bounds.</param>
    /// <param name="max">The maximum corner of the bounds.</param>
    /// <param name="maxItemCount">The maximum number of items to allow per-node before that node subdivides.</param>
    __device__ Octree( const Vector3& min, const Vector3& max, uint_t maxItemCount );

    /// <summary>
    /// Destroys this octree.
    /// </summary>
    __device__ ~Octree();

    /// <summary>
    /// Gets this octree's bounds.
    /// </summary>
    __device__ const BoundingBox& GetBounds() const;

    /// <summary>
    /// Queries this octree for the nearest piece of geometry that a given ray intersects.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to the piece of geometry.</param>
    /// <param name="sp">The shade point data.</param>
    __device__ const Geometry* QueryIntersections( const Ray& ray, real_t& dist, ShadePoint& sp ) const;

    /// <summary>
    /// Queries this octree to see if the given shadow ray intersects anything.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to collision.</param>
    __device__ bool QueryShadowRay( const Ray& ray, real_t& dist ) const;

    /// <summary>
    /// Adds the given bounding box to this octree.
    /// </summary>
    /// <param name="geometry">The piece of geometry to add.</param>
    __device__ bool Add( const Geometry* geometry );
};

REX_NS_END