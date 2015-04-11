#pragma once

#include "../../Math/BoundingBox.hxx"
#include <vector>

REX_NS_BEGIN

class Geometry;

/// <summary>
/// Defines a pairing between a bounding box and geometry.
/// </summary>
struct BoundsGeometryPair
{
    BoundingBox Bounds;
    const Geometry* Geometry;

    /// <summary>
    /// Creates a new bounding box / geometry pairing.
    /// </summary>
    __host__ BoundsGeometryPair();
};

/// <summary>
/// Defines a device octree.
/// </summary>
struct DeviceOctree
{
    const DeviceOctree* Children[ 8 ];
    BoundingBox         Bounds;
    BoundsGeometryPair* Objects;
    uint32              ObjectCount;

    /// <summary>
    /// Creates a new device octree.
    /// </summary>
    __host__ DeviceOctree();

    /// <summary>
    /// Queries this octree for the nearest piece of geometry that a given ray intersects.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to the piece of geometry.</param>
    __device__ const Geometry* QueryIntersections( const Ray& ray, real64& dist ) const;

    /// <summary>
    /// Queries this octree to see if the given shadow ray intersects anything.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    __device__ bool QueryShadowRay( const Ray& ray ) const;

private:
    /// <summary>
    /// Queries this octree for the nearest piece of geometry that a given ray intersects.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="dist">The distance to the piece of geometry.</param>
    __device__ const Geometry* QueryIntersectionsForReal( const Ray& ray, real64& dist ) const;
};

/// <summary>
/// Defines an octree meant for spatially partitioning static objects based on their bounding boxes.
/// </summary>
class Octree
{
    REX_NONCOPYABLE_CLASS( Octree );

    friend DeviceOctree;

    BoundingBox                     _bounds;
    const uint32                    _countBeforeSubivide;
    bool                            _isDevicePointerStale;
    Octree*                         _hChildren[ 8 ];
    std::vector<BoundsGeometryPair> _hObjects;
    std::vector<BoundsGeometryPair> _dObjects;
    DeviceOctree*                   _dThis;
    void*                           _dThisObjects;

    /// <summary>
    /// Checks to see if this octree has subdivided.
    /// </summary>
    __host__ bool HasSubdivided() const;

    /// <summary>
    /// Subdivides this octree.
    /// </summary>
    __host__ void Subdivide();

    /// <summary>
    /// Updates the device array.
    /// </summary>
    __host__ void UpdateDeviceArray();

public:
    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="bounds">The octree's bounds.</param>
    __host__ Octree( const BoundingBox& bounds );

    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="min">The minimum corner of the bounds.</param>
    /// <param name="max">The maximum corner of the bounds.</param>
    __host__ Octree( const Vector3& min, const Vector3& max );

    /// <summary>
    /// Creates a new octree.
    /// </summary>
    /// <param name="min">The minimum corner of the bounds.</param>
    /// <param name="max">The maximum corner of the bounds.</param>
    /// <param name="maxItemCount">The maximum number of items to allow per-node before that node subdivides.</param>
    __host__ Octree( const Vector3& min, const Vector3& max, uint32 maxItemCount );

    /// <summary>
    /// Destroys this octree.
    /// </summary>
    __host__ ~Octree();

    /// <summary>
    /// Gets this octree's bounds.
    /// </summary>
    __host__ const BoundingBox& GetBounds() const;

    /// <summary>
    /// Adds the given bounding box to this octree.
    /// </summary>
    /// <param name="geometry">The piece of geometry to add.</param>
    __host__ bool Add( const Geometry* geometry );

    /// <summary>
    /// Gets this octree on the device.
    /// </summary>
    __host__ const DeviceOctree* GetOnDevice();
};

REX_NS_END