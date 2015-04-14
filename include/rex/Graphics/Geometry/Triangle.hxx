#pragma once

#include "Geometry.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a triangle.
/// </summary>
class Triangle : public Geometry
{
    REX_IMPLEMENT_DEVICE_MEM_OPS()

    Vector3 _p1;
    Vector3 _p2;
    Vector3 _p3;

public:
    /// <summary>
    /// Creates a new triangle.
    /// </summary>
    /// <param name="material">The material to use with this triangle.</param>
    template<typename T> __device__ Triangle( const T& material );

    /// <summary>
    /// Creates a new triangle.
    /// </summary>
    /// <param name="material">The material to use with this triangle.</param>
    /// <param name="p1">The first point in this triangle.</param>
    /// <param name="p2">The second point in this triangle.</param>
    /// <param name="p3">The third point in this triangle.</param>
    template<typename T> __device__ Triangle( const T& material, const Vector3& p1, const Vector3& p2, const Vector3& p3 );

    /// <summary>
    /// Destroys this triangle.
    /// </summary>
    __device__ virtual ~Triangle();

    /// <summary>
    /// Gets this triangle's bounds.
    /// </summary>
    __device__ virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Gets this triangle's normal.
    /// </summary>
    __device__ Vector3 GetNormal() const;

    /// <summary>
    /// Checks to see if the given ray hits this triangle. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    __device__ virtual bool Hit( const Ray& ray, real_t& tmin, ShadePoint& sp ) const;

    /// <summary>
    /// Performs the same thing as a normal ray hit, but for shadow rays.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    __device__ virtual bool ShadowHit( const Ray& ray, real_t& tmin ) const;
};

REX_NS_END

#include "Triangle.inl"