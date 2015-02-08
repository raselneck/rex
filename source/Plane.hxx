#ifndef __REX_PLANE_HXX
#define __REX_PLANE_HXX
#pragma once

#include "Config.hxx"
#include "Geometry.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a plane as a geometric object.
/// </summary>
struct Plane : public Geometry
{
    Vector3 Point;
    Vector3 Normal;

    /// <summary>
    /// Creates a new plane.
    /// </summary>
    Plane();

    /// <summary>
    /// Creates a new plane.
    /// </summary>
    /// <param name="point">The point through which the plane passes.</param>
    /// <param name="normal">The normal representing the plane.</param>
    Plane( const Vector3& point, const Vector3& normal );

    /// <summary>
    /// Destroys this plane.
    /// </summary>
    ~Plane();

    /// <summary>
    /// Checks to see if the given ray hits this plane. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const;
};

REX_NS_END

#endif