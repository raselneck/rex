#include <rex/Math/Vector2.hxx>
#include <math.h>

REX_NS_BEGIN

// create 2D vector
Vector2::Vector2()
    : Vector2( 0.0, 0.0 )
{
}

// create 2D vector
Vector2::Vector2( real_t all )
    : Vector2( all, all )
{
}

// create 2D vector
Vector2::Vector2( real_t x, real_t y )
    : X( x ),
      Y( y )
{
}

// destroy 2D vector
Vector2::~Vector2()
{
    X = Y = 0.0;
}

// get length
real_t Vector2::Length() const
{
    return std::sqrt( LengthSq() );
}

// get length squared
real_t Vector2::LengthSq() const
{
    return ( X * X ) + ( Y * Y );
}

// normalize a vector
Vector2 Vector2::Normalize( const Vector2& vec )
{
    real_t invlen = real_t( 1.0 ) / vec.Length();
    return Vector2( vec.X * invlen,
                    vec.Y * invlen );
}

// check for equality
bool Vector2::operator==( const Vector2& v ) const
{
    return ( X == v.X )
        && ( Y == v.Y );
}

// check for inequality
bool Vector2::operator!=( const Vector2& v ) const
{
    return ( X != v.X )
        || ( Y != v.Y );
}

// add two vectors
Vector2 Vector2::operator+( const Vector2& v ) const
{
    return Vector2( X + v.X,
                    Y + v.Y );
}

// subtract two vectors
Vector2 Vector2::operator-( const Vector2& v ) const
{
    return Vector2( X - v.X,
                    Y - v.Y );
}

// negate a vector
Vector2 Vector2::operator-() const
{
    return Vector2( -X, -Y );
}

// add assign a vector
Vector2& Vector2::operator+=( const Vector2& v )
{
    X += v.X;
    Y += v.Y;
    return *this;
}

// subtract assign a vector
Vector2& Vector2::operator-=( const Vector2& v )
{
    X -= v.X;
    Y -= v.Y;
    return *this;
}

// multiply assign by a scalar
Vector2& Vector2::operator*=( real_t s )
{
    X *= s;
    Y *= s;
    return *this;
}

// divide assign by a scalar
Vector2& Vector2::operator/=( real_t s )
{
    real_t invs = real_t( 1.0 ) / s;
    X *= invs;
    Y *= invs;
    return *this;
}

// multiply 2D vector and scalar
Vector2 operator*( const Vector2& v, real_t s )
{
    return Vector2( v.X * s,
                    v.Y * s );
}

// multiply 2D vector and scalar
Vector2 operator*( real_t s, const Vector2& v )
{
    return Vector2( v.X * s,
                    v.Y * s );
}

// divide 2D vector and scalar
Vector2 operator/( const Vector2& v, real_t s )
{
    real_t invs = real_t( 1.0 ) / s;
    return Vector2( v.X * invs,
                    v.Y * invs );
}

REX_NS_END