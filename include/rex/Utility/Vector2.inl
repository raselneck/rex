#ifndef __REX_VECTOR2_INL
#define __REX_VECTOR2_INL

#include "Vector2.hxx"
#include "Math.hxx"

REX_NS_BEGIN

// get length
inline real64 Vector2::Length() const
{
    return sqrt( LengthSq() );
}

// get length squared
inline real64 Vector2::LengthSq() const
{
    return X * X + Y * Y;
}

// normalize a vector
inline Vector2 Vector2::Normalize( const Vector2& vec )
{
    real64 invlen = 1.0 / vec.Length();
    return Vector2(
        vec.X * invlen,
        vec.Y * invlen
        );
}

// check for equality
inline bool Vector2::operator==( const Vector2& v ) const
{
    return X == v.X
        && Y == v.Y;
}

// check for inequality
inline bool Vector2::operator!=( const Vector2& v ) const
{
    return X != v.X
        && Y != v.Y;
}

// add two vectors
inline Vector2 Vector2::operator+( const Vector2& v ) const
{
    return Vector2(
        X + v.X,
        Y + v.Y
        );
}

// subtract two vectors
inline Vector2 Vector2::operator-( const Vector2& v ) const
{
    return Vector2(
        X - v.X,
        Y - v.Y
        );
}

// negate a vector
inline Vector2 Vector2::operator-( ) const
{
    return Vector2( -X, -Y );
}

// add assign a vector
inline Vector2& Vector2::operator+=( const Vector2& v )
{
    X += v.X;
    Y += v.Y;
    return *this;
}

// subtract assign a vector
inline Vector2& Vector2::operator-=( const Vector2& v )
{
    X -= v.X;
    Y -= v.Y;
    return *this;
}

// multiply assign by a scalar
inline Vector2& Vector2::operator*=( real64 s )
{
    X *= s;
    Y *= s;
    return *this;
}

// divide assign by a scalar
inline Vector2& Vector2::operator/=( real64 s )
{
    real64 invs = 1.0 / s;
    X *= invs;
    Y *= invs;
    return *this;
}

// multiply 2D vector and scalar
inline Vector2 operator*( const Vector2& v, real64 s )
{
    return Vector2(
        v.X * s,
        v.Y * s
        );
}

// multiply 2D vector and scalar
inline Vector2 operator*( real64 s, const Vector2& v )
{
    return Vector2(
        v.X * s,
        v.Y * s
        );
}

// divide 2D vector and scalar
inline Vector2 operator/( const Vector2& v, real64 s )
{
    real64 invs = 1.0 / s;
    return Vector2(
        v.X * invs,
        v.Y * invs
        );
}

// divide 2D vector and scalar
inline Vector2 operator/( real64 s, const Vector2& v )
{
    real64 invs = 1.0 / s;
    return Vector2(
        v.X * invs,
        v.Y * invs
        );
}

REX_NS_END

#endif