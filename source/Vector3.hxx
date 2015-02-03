#ifndef __REX_VECTOR3_HXX
#define __REX_VECTOR3_HXX
#pragma once

#include "Config.hxx"

REX_NS_BEGIN

struct Color;

/// <summary>
/// Defines a 3-dimensional vector.
/// </summary>
struct Vector3
{
    real64 X;
    real64 Y;
    real64 Z;

    /// <summary>
    /// Creates a new 3D vector.
    /// </summary>
    Vector3();

    /// <summary>
    /// Creates a new 3D vector.
    /// </summary>
    /// <param name="all">The value to use for all components.</param>
    Vector3( real64 all );

    /// <summary>
    /// Creates a new 3D vector.
    /// </summary>
    /// <param name="x">The initial X component.</param>
    /// <param name="y">The initial Y component.</param>
    /// <param name="z">The initial Z component.</param>
    Vector3( real64 x, real64 y, real64 z );

    /// <summary>
    /// Destroys this 3D vector.
    /// </summary>
    ~Vector3();

    /// <summary>
    /// Gets the length of this 3D vector.
    /// </summary>
    double Length() const;

    /// <summary>
    /// Gets the length squared of this 3D vector.
    /// </summary>
    double LengthSq() const;

    /// <summary>
    /// Normalizes this 3D vector.
    /// </summary>
    void Normalize();

    /// <summary>
    /// Gets the dot product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    static real64 Dot( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Gets the cross product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    static Vector3 Cross( const Vector3& v1, const Vector3& v2 );

    operator Color();

    bool operator==( const Vector3& );
    bool operator!=( const Vector3& );

    Vector3 operator+( const Vector3& );
    Vector3 operator-( const Vector3& );
    Vector3 operator-();

    Vector3& operator+=( const Vector3& );
    Vector3& operator-=( const Vector3& );
    Vector3& operator*=( real64 );
    Vector3& operator/=( real64 );
};

Vector3 operator*( const Vector3&, real64 );
Vector3 operator*( real64, const Vector3& );
Vector3 operator/( const Vector3&, real64 );
Vector3 operator/( real64, const Vector3& );

REX_NS_END

#endif