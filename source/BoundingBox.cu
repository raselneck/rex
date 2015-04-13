#include <rex/Math/BoundingBox.hxx>
#include <rex/Math/Math.hxx>

REX_NS_BEGIN

// new bounding box
BoundingBox::BoundingBox( const Vector3& min, const Vector3& max )
{
    // ensure min and max
    _min = Vector3::Min( min, max );
    _max = Vector3::Max( min, max );
}

// new bounding box
BoundingBox::BoundingBox( real_t minX, real_t minY, real_t minZ, real_t maxX, real_t maxY, real_t maxZ )
{
    Vector3 min = Vector3( minX, minY, minZ );
    Vector3 max = Vector3( maxX, maxY, maxZ );

    _min = Vector3::Min( min, max );
    _max = Vector3::Max( min, max );
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
bool BoundingBox::Intersects( const Ray& ray, real_t& dist ) const
{
    // adapted from http://gamedev.stackexchange.com/a/18459/46507
    // NOTE : We're assuming the ray direction is a unit vector here

    // get inverse of direction
    Vector3 dirfrac(
        real_t( 1.0 ) / ray.Direction.X,
        real_t( 1.0 ) / ray.Direction.Y,
        real_t( 1.0 ) / ray.Direction.Z
    );

    // get helpers
    real_t t1 = ( _min.X - ray.Origin.X ) * dirfrac.X;
    real_t t2 = ( _max.X - ray.Origin.X ) * dirfrac.X;
    real_t t3 = ( _min.Y - ray.Origin.Y ) * dirfrac.Y;
    real_t t4 = ( _max.Y - ray.Origin.Y ) * dirfrac.Y;
    real_t t5 = ( _min.Z - ray.Origin.Z ) * dirfrac.Z;
    real_t t6 = ( _max.Z - ray.Origin.Z ) * dirfrac.Z;


    real_t tmin = Math::Max( Math::Max( Math::Min( t1, t2 ), Math::Min( t3, t4 ) ), Math::Min( t5, t6 ) );
    real_t tmax = Math::Min( Math::Min( Math::Max( t1, t2 ), Math::Max( t3, t4 ) ), Math::Max( t5, t6 ) );


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

// set box max
void BoundingBox::SetMin( const Vector3& min )
{
    Vector3 oldMin = _min;
    _min = Vector3::Min( _min,   Vector3::Min( min, _max ) );
    _max = Vector3::Max( oldMin, Vector3::Max( min, _max ) );
}

// set box max
void BoundingBox::SetMax( const Vector3& max )
{
    Vector3 oldMax = _max;
    _max = Vector3::Max( _min, Vector3::Max( max, _max   ) );
    _min = Vector3::Min( _min, Vector3::Min( max, oldMax ) );
}

REX_NS_END