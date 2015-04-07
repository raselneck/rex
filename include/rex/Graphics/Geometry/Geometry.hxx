#pragma once

#include "../Materials/Material.hxx"
#include "../../Math/BoundingBox.hxx"

REX_NS_BEGIN

struct ShadePoint;

/// <summary>
/// An enumeration of possible types of geometry.
/// </summary>
enum class GeometryType
{
    Plane,
    Sphere,
    Triangle,
    Mesh
};

/// <summary>
/// Defines the base for all geometry objects.
/// </summary>
class Geometry
{
    REX_NONCOPYABLE_CLASS( Geometry );

    Material* _hMaterial;
    const Material* _dMaterial;

public:
    /// <summary>
    /// Creates a new piece of geometry.
    /// </summary>
    /// <param name="material">The material to use with this piece of geometry.</param>
    template<typename T> __host__ Geometry( const T& material );

    /// <summary>
    /// Destroys this piece of geometry.
    /// </summary>
    __host__ virtual ~Geometry();

    /// <summary>
    /// Gets this piece of geometry's type.
    /// </summary>
    __both__ virtual GeometryType GetType() const = 0;

    /// <summary>
    /// Gets this piece of geometry's bounds.
    /// </summary>
    __host__ virtual BoundingBox GetBounds() const = 0;

    /// <summary>
    /// Gets this geometric object on the device.
    /// </summary>
    __host__ virtual const Geometry* GetOnDevice() const = 0;

    /// <summary>
    /// Checks to see if the given ray hits this geometric object. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    __device__ virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const = 0;

    /// <summary>
    /// Performs the same thing as a normal ray hit, but for shadow rays.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    __device__ virtual bool ShadowHit( const Ray& ray, real64& tmin ) const = 0;

    /// <summary>
    /// Sets this geometry's material.
    /// </summary>
    /// <param name="material">The new material to use with this piece of geometry.</param>
    template<typename T> __host__ void SetMaterial( const T& material );
};

REX_NS_END

#include "Geometry.inl"