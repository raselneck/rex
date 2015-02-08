#include "Plane.hxx"
#include "ShadePoint.hxx"

REX_NS_BEGIN

// create plane
Plane::Plane()
{
}

// create plane
Plane::Plane( const Vector3& point, const Vector3& normal )
    : Point( point ),
      Normal( Vector3::Normalize( normal ) ) // ensures it is a normal
{
}

// destroy plane
Plane::~Plane()
{
}

// check for ray intersection
bool Plane::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // from "Ray Tracing from the Ground Up", page 56

    real64 t = Vector3::Dot( Point - ray.Origin, Normal ) / Vector3::Dot( ray.Direction, Normal );

    // check for intersection
    if ( t > Math::EPSILON )
    {
        tmin = t;
        sp.Normal = Normal;
        sp.HitPoint = ray.Origin + t * ray.Direction;

        return true;
    }
    return false;
}

REX_NS_END