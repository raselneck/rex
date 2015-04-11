#pragma once

#include "Geometry.hxx"
#include "../../Math/Vector3.hxx"

// TODO : Make get and set methods for the radius and center

REX_NS_BEGIN

/// <summary>
/// Defines a sphere.
/// </summary>
class Sphere : public Geometry
{
    Vector3 _center;
    real64  _radius;
    real64  _invRadius;

protected:
    /// <summary>
    /// Handles when this sphere's material is changed.
    /// </summary>
    virtual void OnChangeMaterial();

public:
    /// <summary>
    /// Creates a new sphere.
    /// </summary>
    /// <param name="material">The material to use with this sphere.</param>
    template<typename T> __host__ Sphere( const T& material );

    /// <summary>
    /// Creates a new sphere.
    /// </summary>
    /// <param name="material">The material to use with this sphere.</param>
    /// <param name="center">The initial center of the sphere.</param>
    /// <param name="radius">The initial radius of the sphere.</param>
    template<typename T> __host__ Sphere( const T& material, const Vector3& center, real64 radius );

    /// <summary>
    /// Destroys this sphere.
    /// </summary>
    __host__ virtual ~Sphere();

    /// <summary>
    /// Gets this sphere's geometry type.
    /// </summary>
    __both__ virtual GeometryType GetType() const;

    /// <summary>
    /// Gets this sphere's bounds.
    /// </summary>
    __host__ virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Gets this sphere on the device.
    /// </summary>
    __host__ virtual const Geometry* GetOnDevice() const;

    /// <summary>
    /// Gets sphere's material on the device.
    /// </summary>
    __device__ virtual const Material* GetDeviceMaterial() const;

    /// <summary>
    /// Checks to see if the given ray hits this sphere. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    __device__ virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const;

    /// <summary>
    /// Performs the same thing as a normal ray hit, but for shadow rays.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    __device__ virtual bool ShadowHit( const Ray& ray, real64& tmin ) const;
};

REX_NS_END

#include "Sphere.inl"