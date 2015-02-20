#include "Plane.hxx"
#include "ShadePoint.hxx"

REX_NS_BEGIN

// new plane
Plane::Plane()
{
}

// new plane
Plane::Plane( const Vector3& point, const Vector3& normal )
    : Plane( point, normal, Color::Black )
{
}

// new plane
Plane::Plane( const Vector3& point, const Vector3& normal, const Color& color  )
    : Geometry( color ),
      _point( point ),
      _normal( Vector3::Normalize( normal ) ) // ensures it is a normal
{
}

// destroy plane
Plane::~Plane()
{
}

// get plane bounds
BoundingBox Plane::GetBounds() const
{
    // this function will never actually be used constructively for planes
    return BoundingBox( Vector3(), Vector3() );
}

// get plane point
const Vector3& Plane::GetPoint() const
{
    return _point;
}

// get plane normal
const Vector3& Plane::GetNormal() const
{
    return _normal;
}

// check for ray intersection
bool Plane::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // from Suffern, 56

    real64 t = Vector3::Dot( _point - ray.Origin, _normal ) / Vector3::Dot( ray.Direction, _normal );

    // check for intersection
    if ( t > Math::EPSILON )
    {
        tmin = t;
        sp.Normal = _normal;
        sp.HitPoint = ray.Origin + t * ray.Direction;

        return true;
    }
    return false;
}

REX_NS_END