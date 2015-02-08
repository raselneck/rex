#include "Sphere.hxx"
#include "ShadePoint.hxx"

REX_NS_BEGIN

// new sphere
Sphere::Sphere()
    : Radius( 0.0 )
{
}

// new sphere
Sphere::Sphere( const Vector3& center, real64 radius )
    : Center( center ),
      Radius( radius )
{
}

// destroy sphere
Sphere::~Sphere()
{
    Radius = 0.0;
}

// check for ray intersection
bool Sphere::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // from "Ray Tracing from the Ground Up", page 58

    // this is basically using the quadratic equation solved for x where x = t
    real64  t    = 0.0;
    Vector3 temp = ray.Origin - Center;
    real64  a    = Vector3::Dot( ray.Direction, ray.Direction );
    real64  b    = 2.0 * Vector3::Dot( temp, ray.Direction );
    real64  c    = Vector3::Dot( temp, temp ) - Radius * Radius;
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
        sp.Normal = ( temp + t * ray.Direction ) / Radius;
        sp.HitPoint = ray.Origin + t * ray.Direction;

        return true;
    }

    // now we need to check the larger root (b^2 + 4ac)
    t = ( -b + e ) * denom;
    if ( t > Math::EPSILON )
    {
        tmin = t;
        sp.Normal = ( temp + t * ray.Direction ) / Radius;
        sp.HitPoint = ray.Origin + t * ray.Direction;

        return true;
    }

    return false;
}

REX_NS_END