#include <rex/Geometry/Plane.hxx>
#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

// create plane
Plane::Plane()
{
}

// create plane
Plane::Plane( const Vector3& point, const Vector3& normal )
    : _point( point ), _normal( normal )
{
}

// destroy plane
Plane::~Plane()
{
}

// get plane bounds
BoundingBox Plane::GetBounds() const
{
    // TODO : Even if we need to use massive bounds, we should probably actually calculate this
    return BoundingBox( Vector3(), Vector3() );
}

// get plane normal
const Vector3& Plane::GetNormal() const
{
    return _normal;
}

// get plane point
const Vector3& Plane::GetPoint() const
{
    return _point;
}

// get sphere geometry type
GeometryType Plane::GetType() const
{
    return GeometryType::Plane;
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
        sp.LocalHitPoint = ray.Origin + t * ray.Direction;

        return true;
    }
    return false;
}

// check for shadow ray intersection
bool Plane::ShadowHit( const Ray& ray, real64& tmin ) const
{
    // from Suffern, 301

    real64 t = Vector3::Dot( _point - ray.Origin, _normal ) / Vector3::Dot( ray.Direction, _normal );

    // check for intersection
    if ( t > Math::EPSILON )
    {
        tmin = t;
        return true;
    }
    return false;
}

REX_NS_END