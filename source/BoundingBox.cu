#include <rex/Math/BoundingBox.hxx>
#include <rex/Math/Math.hxx>

REX_NS_BEGIN

// new bounding box
BoundingBox::BoundingBox( const vec3& min, const vec3& max )
    : _min( min ),
      _max( max )
{
}

// new bounding box
BoundingBox::BoundingBox( real32 minX, real32 minY, real32 minZ, real32 maxX, real32 maxY, real32 maxZ )
    : _min( minX, minY, minZ ),
      _max( maxX, maxY, maxZ )
{
}

// destroy bounding box
BoundingBox::~BoundingBox()
{
}

// check for containment type
ContainmentType BoundingBox::Contains( const BoundingBox& bbox ) const
{
    if ( bbox._max.x < _min.x ||
         bbox._min.x > _max.x ||
         bbox._max.y < _min.y ||
         bbox._min.y > _max.y ||
         bbox._max.z < _min.z ||
         bbox._min.z > _max.z )
    {
        return ContainmentType::Disjoint;
    }


    if ( bbox._min.x >= _min.x &&
         bbox._max.x <= _max.x &&
         bbox._min.y >= _min.y &&
         bbox._max.y <= _max.y &&
         bbox._min.z >= _min.z &&
         bbox._max.z <= _max.z )
    {
        return ContainmentType::Contains;
    }

    return ContainmentType::Intersects;
}

// get center
vec3 BoundingBox::GetCenter() const
{
    return _min + GetSize() * 0.5f;
}

// get bounding box min
const vec3& BoundingBox::GetMin() const
{
    return _min;
}

// get bounding box max
const vec3& BoundingBox::GetMax() const
{
    return _max;
}

// get size
vec3 BoundingBox::GetSize() const
{
    return _max - _min;
}

// check for ray-box intersection
bool BoundingBox::Intersects( const Ray& ray, real32& dist ) const
{
    // adapted from http://gamedev.stackexchange.com/a/18459/46507
    // NOTE : We're assuming the ray direction is a unit vector here

    // get inverse of direction
    vec3 dirfrac = 1.0f / ray.Direction;

    // get helpers
    real32 t1 = ( _min.x - ray.Origin.x ) * dirfrac.x;
    real32 t2 = ( _max.x - ray.Origin.x ) * dirfrac.x;
    real32 t3 = ( _min.y - ray.Origin.y ) * dirfrac.y;
    real32 t4 = ( _max.y - ray.Origin.y ) * dirfrac.y;
    real32 t5 = ( _min.z - ray.Origin.z ) * dirfrac.z;
    real32 t6 = ( _max.z - ray.Origin.z ) * dirfrac.z;


    real32 tmin = Math::Max( Math::Max( Math::Min( t1, t2 ), Math::Min( t3, t4 ) ), Math::Min( t5, t6 ) );
    real32 tmax = Math::Min( Math::Min( Math::Max( t1, t2 ), Math::Max( t3, t4 ) ), Math::Max( t5, t6 ) );


    // if tmax < 0, ray (line) is intersecting box, but whole box is behind us
    if ( tmax < 0 )
    {
        dist = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect box
    if ( tmin > tmax )
    {
        dist = tmax;
        return false;
    }

    dist = tmin;
    return true;
}

REX_NS_END