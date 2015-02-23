#ifndef __REX_RAY_HXX
#define __REX_RAY_HXX

#include "../Config.hxx"
#include "Vector3.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a ray.
/// </summary>
struct Ray
{
    Vector3 Origin;
    Vector3 Direction;

    /// <summary>
    /// Creates a new ray.
    /// <summary>
    Ray();

    /// <summary>
    /// Creates a new ray.
    /// <summary>
    /// <param name="origin">The ray's origin.</param>
    /// <param name="direction">The ray's direction.</param>
    Ray( const Vector3& origin, const Vector3& direction );

    /// <summary>
    /// Destroys this ray.
    /// <summary>
    ~Ray();
};

REX_NS_END

#endif