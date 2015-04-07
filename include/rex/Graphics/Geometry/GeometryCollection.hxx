#pragma once

#include "Geometry.hxx"
#include "Sphere.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines a geometry collection.
/// </summary>
class GeometryCollection
{
    REX_NONCOPYABLE_CLASS( GeometryCollection );

    std::vector<Geometry*>       _hGeometry;
    std::vector<const Geometry*> _dGeometry;
    Geometry**                   _dGeometryArray;

    /// <summary>
    /// Updates the device array.
    /// </summary>
    __host__ void UpdateDeviceArray();

public:
    /// <summary>
    /// Creates a new geometry collection.
    /// </summary>
    __host__ GeometryCollection();

    /// <summary>
    /// Destroys geometry collection.
    /// </summary>
    __host__ ~GeometryCollection();

    /// <summary>
    /// Gets the number of geometric objects in this collection.
    /// </summary>
    __host__ uint32 GetGeometryCount() const;

    /// <summary>
    /// Gets the array of geometric objects in this collection.
    /// </summary>
    __host__ const std::vector<Geometry*>& GetGeometry() const;

    /// <summary>
    /// Gets the array of geometric objects in this collection that is on the device.
    /// </summary>
    __host__ const Geometry** GetDeviceGeometry() const;

    /// <summary>
    /// Adds a sphere to this geometry collection.
    /// </summary>
    __host__ Sphere* AddSphere();

    /// <summary>
    /// Adds a sphere to this geometry collection.
    /// </summary>
    /// <param name="center">The sphere's initial center.</param>
    /// <param name="radius">The sphere's initial radius.</param>
    __host__ Sphere* AddSphere( const Vector3& center, real64 radius );
};

REX_NS_END