#include <rex/Math/Vector3.hxx>
#include <rex/Math/Math.hxx>

REX_NS_BEGIN

// create 3D vector
Vector3::Vector3()
    : Vector3( 0.0, 0.0, 0.0 )
{
}

// create 3D vector
Vector3::Vector3( real_t all )
    : Vector3( all, all, all )
{
}

// create 3D vector
Vector3::Vector3( real_t x, real_t y, real_t z )
    : X( x ),
      Y( y ),
      Z( z )
{
}

// destroy 3D vector
Vector3::~Vector3()
{
    X = Y = Z = 0.0;
}

// get vector length
real_t Vector3::Length() const
{
    return std::sqrt( LengthSq() );
}

// get vector length squared
real_t Vector3::LengthSq() const
{
    return X * X + Y * Y + Z * Z;
}

// dot product of two vectors
Vector3 Vector3::Cross( const Vector3& v1, const Vector3& v2 )
{
    return Vector3( v1.Y * v2.Z - v1.Z * v2.Y,
                    v1.Z * v2.X - v1.X * v2.Z,
                    v1.X * v2.Y - v1.Y * v2.X );
}

// distance between two vectors
real_t Vector3::Distance( const Vector3& v1, const Vector3& v2 )
{
    Vector3 vec = v2 - v1;
    return vec.Length();
}

// distance squared between two vectors
real_t Vector3::DistanceSq( const Vector3& v1, const Vector3& v2 )
{
    Vector3 vec = v2 - v1;
    return vec.LengthSq();
}

// dot product of two vectors
real_t Vector3::Dot( const Vector3& v1, const Vector3& v2 )
{
    return v1.X * v2.X
         + v1.Y * v2.Y
         + v1.Z * v2.Z;
}

// get min components of two vectors
Vector3 Vector3::Min( const Vector3& v1, const Vector3& v2 )
{
    return Vector3( Math::Min( v1.X, v2.X ),
                    Math::Min( v1.Y, v2.Y ),
                    Math::Min( v1.Z, v2.Z ) );
}

// get max components of two vectors
Vector3 Vector3::Max( const Vector3& v1, const Vector3& v2 )
{
    return Vector3( Math::Max( v1.X, v2.X ),
                    Math::Max( v1.Y, v2.Y ),
                    Math::Max( v1.Z, v2.Z ) );
}

// normalize a vector
Vector3 Vector3::Normalize( const Vector3& vec )
{
    real_t invlen = real_t( 1.0 ) / vec.Length();
    return Vector3( vec.X * invlen,
                    vec.Y * invlen,
                    vec.Z * invlen );
}

bool Vector3::operator==( const Vector3& c ) const
{
    return X == c.X
        && Y == c.Y
        && Z == c.Z;
}

bool Vector3::operator!=( const Vector3& c ) const
{
    return ( X != c.X )
        || ( Y != c.Y )
        || ( Z != c.Z );
}

Vector3 Vector3::operator+( const Vector3& c ) const
{
    return Vector3( X + c.X,
                    Y + c.Y,
                    Z + c.Z );
}

Vector3 Vector3::operator-( const Vector3& c ) const
{
    return Vector3( X - c.X,
                    Y - c.Y,
                    Z - c.Z );
}

Vector3 Vector3::operator-() const
{
    return Vector3( -X, -Y, -Z );
}

Vector3& Vector3::operator+=( const Vector3& c )
{
    X += c.X;
    Y += c.Y;
    Z += c.Z;
    return *this;
}

Vector3& Vector3::operator-=( const Vector3& c )
{
    X -= c.X;
    Y -= c.Y;
    Z -= c.Z;
    return *this;
}

Vector3& Vector3::operator*=( real_t s )
{
    X *= s;
    Y *= s;
    Z *= s;
    return *this;
}

Vector3& Vector3::operator/=( real_t s )
{
    real_t invs = real_t( 1.0 ) / s;
    X *= invs;
    Y *= invs;
    Z *= invs;
    return *this;
}

Vector3 operator*( const Vector3& v, real_t s )
{
    return Vector3( v.X * s,
                    v.Y * s,
                    v.Z * s );
}

Vector3 operator*( real_t s, const Vector3& v )
{
    return Vector3( v.X * s,
                    v.Y * s,
                    v.Z * s );
}

Vector3 operator/( const Vector3& v, real_t s )
{
    real_t invs = real_t( 1.0 ) / s;
    return Vector3( v.X * invs,
                    v.Y * invs,
                    v.Z * invs );
}


REX_NS_END