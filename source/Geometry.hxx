#ifndef __REX_GEOOBJECT_HXX
#define __REX_GEOOBJECT_HXX
#pragma once

#include "Config.hxx"
#include "Color.hxx"
#include "Ray.hxx"

REX_NS_BEGIN

struct ShadePoint;

/// <summary>
/// The base class for geometric objects.
/// </summary>
struct Geometry
{
    Color Color;

    /// <summary>
    /// Creates a new geometric object.
    /// </summary>
    Geometry();

    /// <summary>
    /// Creates a new geometric object.
    /// </summary>
    /// <param name="color">The geometric object's color.</param>
    Geometry( const rex::Color& color );

    /// <summary>
    /// Destroys this geometric object.
    /// </summary>
    virtual ~Geometry();

    /// <summary>
    /// Checks to see if the given ray hits this geometric object. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const = 0;
};

REX_NS_END

#endif