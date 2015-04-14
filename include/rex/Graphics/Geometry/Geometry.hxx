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
    Sphere,
    Triangle,
    Mesh
};

/// <summary>
/// Defines the base for all geometry objects.
/// </summary>
class Geometry
{
    REX_NONCOPYABLE_CLASS( Geometry )
    REX_IMPLEMENT_DEVICE_MEM_OPS()

protected:
    Material*           _material;
    const GeometryType  _geometryType;

public:
    /// <summary>
    /// Creates a new piece of geometry.
    /// </summary>
    /// <param name="type">The type of this geometry.</param>
    /// <param name="material">The material to use with this piece of geometry.</param>
    template<typename T> __device__ Geometry( GeometryType type, const T& material );

    /// <summary>
    /// Destroys this piece of geometry.
    /// </summary>
    __device__ virtual ~Geometry();

    /// <summary>
    /// Gets this piece of geometry's bounds.
    /// </summary>
    __device__ virtual BoundingBox GetBounds() const = 0;

    /// <summary>
    /// Gets this geometric object's material.
    /// </summary>
    __device__ const Material* GetMaterial() const;

    /// <summary>
    /// Gets this piece of geometry's type.
    /// </summary>
    __device__ GeometryType GetType() const;

    /// <summary>
    /// Checks to see if the given ray hits this geometric object. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    __device__ virtual bool Hit( const Ray& ray, real_t& tmin, ShadePoint& sp ) const = 0;

    /// <summary>
    /// Performs the same thing as a normal ray hit, but for shadow rays.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    __device__ virtual bool ShadowHit( const Ray& ray, real_t& tmin ) const = 0;

    /// <summary>
    /// Sets this geometry's material.
    /// </summary>
    /// <param name="material">The new material to use with this piece of geometry.</param>
    template<typename T> __device__ void SetMaterial( const T& material );
};

REX_NS_END

#include "Geometry.inl"