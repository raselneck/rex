#include "BoundingBox.hxx"

REX_NS_BEGIN

// new bounding box
BoundingBox::BoundingBox( const Vector3& min, const Vector3& max )
    : _min( min ), _max( max )
{
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

REX_NS_END