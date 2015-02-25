#include <rex/Geometry/BoundingBox.hxx>

REX_NS_BEGIN

// new bounding box
BoundingBox::BoundingBox( const Vector3& min, const Vector3& max )
{
    // ensure min and max
    _min = Vector3::Min( min, max );
    _max = Vector3::Max( min, max );
}

// new bounding box
BoundingBox::BoundingBox( real64 minX, real64 minY, real64 minZ, real64 maxX, real64 maxY, real64 maxZ )
    : _min( minX, minY, minZ ), _max( maxX, maxY, maxZ )
{
}

// destroy bounding box
BoundingBox::~BoundingBox()
{
}

// check for containment type
ContainmentType BoundingBox::Contains( const BoundingBox& bbox ) const
{
    if ( bbox._max.X < _min.X ||
         bbox._min.X > _max.X ||
         bbox._max.Y < _min.Y ||
         bbox._min.Y > _max.Y ||
         bbox._max.Z < _min.Z ||
         bbox._min.Z > _max.Z )
    {
        return ContainmentType::Disjoint;
    }


    if ( bbox._min.X >= _min.X &&
         bbox._max.X <= _max.X &&
         bbox._min.Y >= _min.Y &&
         bbox._max.Y <= _max.Y &&
         bbox._min.Z >= _min.Z &&
         bbox._max.Z <= _max.Z )
    {
        return ContainmentType::Contains;
    }

    return ContainmentType::Intersects;
}

// get center
Vector3 BoundingBox::GetCenter() const
{
    return _min + GetSize() * 0.5;
}

// get bounding box min
const Vector3& BoundingBox::GetMin() const
{
    return _min;
}

// get bounding box max
const Vector3& BoundingBox::GetMax() const
{
    return _max;
}

// get size
Vector3 BoundingBox::GetSize() const
{
    return _max - _min;
}

// check for ray-box intersection
bool BoundingBox::Intersects( const Ray& ray, real32& dist ) const
{
    // adapted from http://gamedev.stackexchange.com/a/18459/46507
    // NOTE : We're assuming the ray direction is a unit vector here

    // get inverse of direction
    Vector3 dirfrac;
    dirfrac.X = 1.0 / ray.Direction.X;
    dirfrac.Y = 1.0 / ray.Direction.Y;
    dirfrac.Z = 1.0 / ray.Direction.Z;

    // get helpers
    real64 t1 = ( _min.X - ray.Origin.X ) * dirfrac.X;
    real64 t2 = ( _max.X - ray.Origin.X ) * dirfrac.X;
    real64 t3 = ( _min.Y - ray.Origin.Y ) * dirfrac.Y;
    real64 t4 = ( _max.Y - ray.Origin.Y ) * dirfrac.Y;
    real64 t5 = ( _min.Z - ray.Origin.Z ) * dirfrac.Z;
    real64 t6 = ( _max.Z - ray.Origin.Z ) * dirfrac.Z;


    real64 tmin = Math::Max( Math::Max( Math::Min( t1, t2 ), Math::Min( t3, t4 ) ), Math::Min( t5, t6 ) );
    real64 tmax = Math::Min( Math::Min( Math::Max( t1, t2 ), Math::Max( t3, t4 ) ), Math::Max( t5, t6 ) );


    // if tmax < 0, ray (line) is intersecting box, but whole box is behind us
    if ( tmax < 0 )
    {
        dist = static_cast<real32>( tmax );
        return false;
    }

    // if tmin > tmax, ray doesn't intersect box
    if ( tmin > tmax )
    {
        dist = static_cast<real32>( tmax );
        return false;
    }

    dist = static_cast<real32>( tmin );
    return true;
}

REX_NS_END