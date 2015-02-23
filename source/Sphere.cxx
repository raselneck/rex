#include <rex/Geometry/Sphere.hxx>
#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

// create sphere
Sphere::Sphere()
    : Sphere( Vector3(), 0.0 )
{
}

// create sphere
Sphere::Sphere( const Vector3& center, real64 radius )
    : _center( center ), _radius( radius )
{
    _invRadius = 1.0 / _radius;
}

// destroy sphere
Sphere::~Sphere()
{
    _radius = 0.0;
}

// get sphere's bounds
BoundingBox Sphere::GetBounds() const
{
    Vector3 size( _radius );
    return BoundingBox( _center - size, _center + size );
}

// get sphere center
const Vector3& Sphere::GetCenter() const
{
    return _center;
}

// get sphere radius
real64 Sphere::GetRadius() const
{
    return _radius;
}

// get sphere geometry type
GeometryType Sphere::GetType() const
{
    return GeometryType::Sphere;
}

// check for ray intersection
bool Sphere::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // from "Ray Tracing from the Ground Up", page 58

    // this is basically using the quadratic equation solved for x where x = t
    real64  t    = 0.0;
    Vector3 temp = ray.Origin - _center;
    real64  a    = Vector3::Dot( ray.Direction, ray.Direction );
    real64  b    = 2.0 * Vector3::Dot( temp, ray.Direction );
    real64  c    = Vector3::Dot( temp, temp ) - _radius * _radius;
    real64  disc = b * b - 4.0 * a * c; // discriminant

    // check if the ray misses completely
    if ( disc < 0.0 )
    {
        return false;
    }

    // now we need to check the smaller root (b^2 - 4ac)
    real64 e     = sqrt( disc );
    real64 denom = 1.0 / ( 2.0 * a );
    t = ( -b - e ) * denom;
    if ( t > Math::EPSILON )
    {
        tmin = t;
        sp.Normal = ( temp + t * ray.Direction ) * _invRadius;
        sp.LocalHitPoint = ray.Origin + t * ray.Direction;

        return true;
    }

    // now we need to check the larger root (b^2 + 4ac)
    t = ( -b + e ) * denom;
    if ( t > Math::EPSILON )
    {
        tmin = t;
        sp.Normal = ( temp + t * ray.Direction ) * _invRadius;
        sp.LocalHitPoint = ray.Origin + t * ray.Direction;

        return true;
    }

    return false;
}

REX_NS_END