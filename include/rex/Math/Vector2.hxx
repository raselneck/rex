#pragma once

#include "../Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a 2-dimensional vector.
/// </summary>
struct Vector2
{
    real_t X;
    real_t Y;

    /// <summary>
    /// Creates a new 2D vector.
    /// </summary>
    __both__ Vector2();

    /// <summary>
    /// Creates a new 2D vector.
    /// </summary>
    /// <param name="all">The value to use for all components.</param>
    __both__ Vector2( real_t all );

    /// <summary>
    /// Creates a new 2D vector.
    /// </summary>
    /// <param name="x">The initial X component.</param>
    /// <param name="y">The initial Y component.</param>
    __both__ Vector2( real_t x, real_t y );

    /// <summary>
    /// Destroys this 2D vector.
    /// </summary>
    __both__ ~Vector2();

    /// <summary>
    /// Gets the length of this 2D vector.
    /// </summary>
    __both__ real_t Length() const;

    /// <summary>
    /// Gets the length squared of this 2D vector.
    /// </summary>
    __both__ real_t LengthSq() const;

    
    /// <summary>
    /// Normalizes the given vector.
    /// </summary>
    /// <param name="vec">The vector.</param>
    __both__ static Vector2 Normalize( const Vector2& vec );


    __both__ bool operator==( const Vector2& ) const;
    __both__ bool operator!=( const Vector2& ) const;

    __both__ Vector2 operator+( const Vector2& ) const;
    __both__ Vector2 operator-( const Vector2& ) const;
    __both__ Vector2 operator-() const;

    __both__ Vector2& operator+=( const Vector2& );
    __both__ Vector2& operator-=( const Vector2& );
    __both__ Vector2& operator*=( real_t );
    __both__ Vector2& operator/=( real_t );
};

__both__ Vector2 operator*( const Vector2&, real_t );
__both__ Vector2 operator*( real_t, const Vector2& );
__both__ Vector2 operator/( const Vector2&, real_t );

REX_NS_END