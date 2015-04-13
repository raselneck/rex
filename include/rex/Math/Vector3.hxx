#pragma once

#include "../Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a 3-dimensional vector.
/// </summary>
struct Vector3
{
    real_t X;
    real_t Y;
    real_t Z;

    /// <summary>
    /// Creates a new 3D vector.
    /// </summary>
    __both__ Vector3();

    /// <summary>
    /// Creates a new 3D vector.
    /// </summary>
    /// <param name="all">The value to use for all components.</param>
    __both__ Vector3( real_t all );

    /// <summary>
    /// Creates a new 3D vector.
    /// </summary>
    /// <param name="x">The initial X component.</param>
    /// <param name="y">The initial Y component.</param>
    /// <param name="z">The initial Z component.</param>
    __both__ Vector3( real_t x, real_t y, real_t z );

    /// <summary>
    /// Destroys this 3D vector.
    /// </summary>
    __both__ ~Vector3();

    /// <summary>
    /// Gets the length of this 3D vector.
    /// </summary>
    __both__ real_t Length() const;

    /// <summary>
    /// Gets the length squared of this 3D vector.
    /// </summary>
    __both__ real_t LengthSq() const;

    /// <summary>
    /// Gets the cross product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    __both__ static Vector3 Cross( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Gets the distance between two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    __both__ static real_t Distance( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Gets the distance squared between two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    __both__ static real_t DistanceSq( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Gets the dot product of two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    __both__ static real_t Dot( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Gets a vector containing the minimum components of each given vector.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    __both__ static Vector3 Min( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Gets a vector containing the maximum components of each given vector.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    __both__ static Vector3 Max( const Vector3& v1, const Vector3& v2 );

    /// <summary>
    /// Normalizes the given vector.
    /// </summary>
    /// <param name="vec">The vector.</param>
    __both__ static Vector3 Normalize( const Vector3& vec );

    __both__ bool operator==( const Vector3& ) const;
    __both__ bool operator!=( const Vector3& ) const;

    __both__ Vector3 operator+( const Vector3& ) const;
    __both__ Vector3 operator-( const Vector3& ) const;
    __both__ Vector3 operator-() const;

    __both__ Vector3& operator+=( const Vector3& );
    __both__ Vector3& operator-=( const Vector3& );
    __both__ Vector3& operator*=( real_t );
    __both__ Vector3& operator/=( real_t );
};

__both__ Vector3 operator*( const Vector3&, real_t );
__both__ Vector3 operator*( real_t, const Vector3& );
__both__ Vector3 operator/( const Vector3&, real_t );

REX_NS_END