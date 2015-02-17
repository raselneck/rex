#ifndef __REX_VECTOR2_HXX
#define __REX_VECTOR2_HXX

#include "Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a 2-dimensional vector.
/// </summary>
struct Vector2
{
    real64 X;
    real64 Y;

    /// <summary>
    /// Creates a new 2D vector.
    /// </summary>
    Vector2();

    /// <summary>
    /// Creates a new 2D vector.
    /// </summary>
    /// <param name="all">The value to use for all components.</param>
    Vector2( real64 all );

    /// <summary>
    /// Creates a new 2D vector.
    /// </summary>
    /// <param name="x">The initial X component.</param>
    /// <param name="y">The initial Y component.</param>
    Vector2( real64 x, real64 y );

    /// <summary>
    /// Destroys this 2D vector.
    /// </summary>
    ~Vector2();

    /// <summary>
    /// Gets the length of this 2D vector.
    /// </summary>
    real64 Length() const;

    /// <summary>
    /// Gets the length squared of this 2D vector.
    /// </summary>
    real64 LengthSq() const;

    
    /// <summary>
    /// Normalizes the given vector.
    /// </summary>
    /// <param name="vec">The vector.</param>
    static Vector2 Normalize( const Vector2& vec );

    bool operator==( const Vector2& ) const;
    bool operator!=( const Vector2& ) const;

    Vector2 operator+( const Vector2& ) const;
    Vector2 operator-( const Vector2& ) const;
    Vector2 operator-() const;

    Vector2& operator+=( const Vector2& );
    Vector2& operator-=( const Vector2& );
    Vector2& operator*=( real64 );
    Vector2& operator/=( real64 );
};

Vector2 operator*( const Vector2&, real64 );
Vector2 operator*( real64, const Vector2& );
Vector2 operator/( const Vector2&, real64 );
Vector2 operator/( real64, const Vector2& );

REX_NS_END

#include "Vector2.inl"
#endif