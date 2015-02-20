#ifndef __REX_VECTOR3_INL
#define __REX_VECTOR3_INL

#include "Vector3.hxx"
#include "Math.hxx"

REX_NS_BEGIN

// get vector length
inline real64 Vector3::Length() const
{
    return sqrt( LengthSq() );
}

// get vector length squared
inline real64 Vector3::LengthSq() const
{
    return X * X + Y * Y + Z * Z;
}

// dot product of two vectors
inline real64 Vector3::Dot( const Vector3& v1, const Vector3& v2 )
{
    return v1.X * v2.X
        + v1.Y * v2.Y
        + v1.Z * v2.Z;
}

// dot product of two vectors
inline Vector3 Vector3::Cross( const Vector3& v1, const Vector3& v2 )
{
    return Vector3(
        v1.Y * v2.Z - v1.Z * v2.Y,
        v1.Z * v2.X - v1.X * v2.Z,
        v1.X * v2.Y - v1.Y * v2.X
        );
}

// get min components of two vectors
inline Vector3 Vector3::Min( const Vector3& v1, const Vector3& v2 )
{
    return Vector3(
        Math::Min( v1.X, v2.X ),
        Math::Min( v1.Y, v2.Y ),
        Math::Min( v1.Z, v2.Z )
    );
}

// get max components of two vectors
inline Vector3 Vector3::Max( const Vector3& v1, const Vector3& v2 )
{
    return Vector3(
        Math::Max( v1.X, v2.X ),
        Math::Max( v1.Y, v2.Y ),
        Math::Max( v1.Z, v2.Z )
    );
}

// normalize a vector
inline Vector3 Vector3::Normalize( const Vector3& vec )
{
    real64 invlen = 1.0 / vec.Length();
    return Vector3(
        vec.X * invlen,
        vec.Y * invlen,
        vec.Z * invlen
        );
}

inline bool Vector3::operator==( const Vector3& c ) const
{
    return X == c.X
        && Y == c.Y
        && Z == c.Z;
}

inline bool Vector3::operator!=( const Vector3& c ) const
{
    return X != c.X
        && Y != c.Y
        && Z != c.Z;
}

inline Vector3 Vector3::operator+( const Vector3& c ) const
{
    return Vector3(
        X + c.X,
        Y + c.Y,
        Z + c.Z
        );
}

inline Vector3 Vector3::operator-( const Vector3& c ) const
{
    return Vector3(
        X - c.X,
        Y - c.Y,
        Z - c.Z
        );
}

inline Vector3 Vector3::operator-( ) const
{
    return Vector3( -X, -Y, -Z );
}

inline Vector3& Vector3::operator+=( const Vector3& c )
{
    X += c.X;
    Y += c.Y;
    Z += c.Z;
    return *this;
}

inline Vector3& Vector3::operator-=( const Vector3& c )
{
    X -= c.X;
    Y -= c.Y;
    Z -= c.Z;
    return *this;
}

inline Vector3& Vector3::operator*=( real64 s )
{
    X *= s;
    Y *= s;
    Z *= s;
    return *this;
}

inline Vector3& Vector3::operator/=( real64 s )
{
    real64 invs = 1.0 / s;
    X *= invs;
    Y *= invs;
    Z *= invs;
    return *this;
}

inline Vector3 operator*( const Vector3& v, real64 s )
{
    return Vector3(
        v.X * s,
        v.Y * s,
        v.Z * s
        );
}

inline Vector3 operator*( real64 s, const Vector3& v )
{
    return Vector3(
        v.X * s,
        v.Y * s,
        v.Z * s
        );
}

inline Vector3 operator/( const Vector3& v, real64 s )
{
    real64 invs = 1.0 / s;
    return Vector3(
        v.X * invs,
        v.Y * invs,
        v.Z * invs
        );
}

REX_NS_END

#endif