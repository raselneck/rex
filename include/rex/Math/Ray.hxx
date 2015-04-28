#pragma once

#include "../Config.hxx"
#include "Math.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a ray.
/// </summary>
struct Ray
{
    vec3 Origin;
    vec3 Direction;

    /// <summary>
    /// Creates a new ray.
    /// <summary>
    __both__ Ray();

    /// <summary>
    /// Creates a new ray.
    /// <summary>
    /// <param name="origin">The ray's origin.</param>
    /// <param name="direction">The ray's direction.</param>
    __both__ Ray( const vec3& origin, const vec3& direction );

    /// <summary>
    /// Destroys this ray.
    /// <summary>
    __both__ ~Ray();
};

REX_NS_END