#include <rex/Graphics/Geometry/Octree.hxx>
#include <rex/Graphics/Geometry/Geometry.hxx>
#include <rex/Graphics/Geometry/Sphere.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Math/Math.hxx>
#include <rex/Utility/GC.hxx>
#include <rex/Utility/Logger.hxx>


#define DEFAULT_MAX_ITEM_COUNT 12
#include <stdio.h>


REX_NS_BEGIN

// create a new bounding box / geometry pair
__device__ BoundsGeometryPair::BoundsGeometryPair()
    : Bounds( vec3(), vec3() )
{
}

// create an octree w/ bounds
__device__ Octree::Octree( const BoundingBox& bounds )
    : Octree( bounds, DEFAULT_MAX_ITEM_COUNT )
{
}

// create an octree w/ min and max corner
__device__ Octree::Octree( const vec3& min, const vec3& max )
    : Octree( BoundingBox( min, max ), DEFAULT_MAX_ITEM_COUNT )
{
}

// create an octree w/ bounds and max item count
__device__ Octree::Octree( const BoundingBox& bounds, uint32 maxItemCount )
    : _bounds( bounds )
    , _countBeforeSubivide( maxItemCount )
{
    // avoid a loop but clear out the children
    _children[ 0 ] = nullptr;
    _children[ 1 ] = nullptr;
    _children[ 2 ] = nullptr;
    _children[ 3 ] = nullptr;
    _children[ 4 ] = nullptr;
    _children[ 5 ] = nullptr;
    _children[ 6 ] = nullptr;
    _children[ 7 ] = nullptr;
}

// create an octree w/ min corner, max corner, and max item count
__device__ Octree::Octree( const vec3& min, const vec3& max, uint32 maxItemCount )
    : Octree( BoundingBox( min, max ), maxItemCount )
{
}

// destroy this octree
__device__ Octree::~Octree()
{
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            delete _children[ i ];
            _children[ i ] = nullptr;
        }
    }
}

// get the octree's bounds
__device__ const BoundingBox& Octree::GetBounds() const
{
    return _bounds;
}

// check if this octree has subdivided
__device__ bool Octree::HasSubdivided() const
{
    return _children[ 0 ] != nullptr;
}

// query the intersections of the given ray and return the closest hit object
__device__ const Geometry* Octree::QueryIntersections( const Ray& ray, real32& dist, ShadePoint& sp ) const
{
    // reset the distance
    dist = Math::HugeValue();


    // make sure the ray even intersects us
    real32 tempDist = 0.0;
    if ( _bounds.Intersects( ray, tempDist ) )
    {
        return QueryIntersectionsForReal( ray, dist, sp );
    }


    // if we don't, then return nothing
    return nullptr;
}

// queries the intersections for real this time
__device__ const Geometry* Octree::QueryIntersectionsForReal( const Ray& ray, real32& dist, ShadePoint& sp ) const
{
    const Geometry* closest   = nullptr;
    real32          tempDist  = 0.0;
    ShadePoint      tempPoint = sp;

    // check our children first
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            // have the child query for ray intersections
            const Octree*   child = _children[ i ];
            const Geometry* geom  = child->QueryIntersections( ray, tempDist, tempPoint );

            // check to see if the child intersected the ray
            if ( ( geom != nullptr ) && ( tempDist < dist ) )
            {
                closest = geom;
                dist    = tempDist;
                sp      = tempPoint;
            }
        }
    }

    // now check our objects
    for ( uint32 i = 0; i < _objects.GetSize(); ++i )
    {
        // TODO : Would just checking for the hit be faster than doing the bounds intersect first?
        BoundsGeometryPair pair = _objects[ i ];
        if ( pair.Bounds.Intersects( ray, tempDist ) && ( tempDist < dist ) )
        {
            if ( pair.Geometry->Hit( ray, tempDist, tempPoint ) && ( tempDist < dist ) )
            {
                closest = pair.Geometry;
                dist    = tempDist;
                sp      = tempPoint;
            }
        }
    }

    return closest;
}

// queries the intersections of the given ray for shadows
__device__ bool Octree::QueryShadowRay( const Ray& ray, real32& dist ) const
{
    real32 d = 0.0;
    bool hit = false;
    dist     = Math::HugeValue();

    // ensure the ray even intersects our bounds
    if ( !_bounds.Intersects( ray, d ) )
    {
        return false;
    }

    // check the children first if we have subdivided
    // TODO : Will this work? Should we still check our objects?
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            if ( _children[ i ]->QueryShadowRay( ray, d ) && ( d < dist ) )
            {
                hit  = true;
                dist = d;
            }
        }
    }

    // now check all of our objects
    uint32 count = _objects.GetSize();
    for ( uint32 i = 0; i < count; ++i )
    {
        const Geometry* geom = _objects[ i ].Geometry;
        if ( geom->ShadowHit( ray, d ) && ( d < dist ) )
        {
            hit  = true;
            dist = d;
        }
    }

    return hit;
}

// add the given piece of geometry to this octree
__device__ bool Octree::Add( const Geometry* geometry )
{
    return Add( geometry, geometry->GetBounds() );
}

// add the given piece of geometry to this octree
__device__ bool Octree::Add( const Geometry* geometry, const BoundingBox& bounds )
{
    // ensure we were given a valid piece of geometry
    if ( !geometry )
    {
        return false;
    }

    // make sure we contain the geometry's bounding box
    ContainmentType ctype = _bounds.Contains( bounds );
    if ( ctype != ContainmentType::Contains )
    {
        return false;
    }


    // create the pair
    BoundsGeometryPair pair;
    pair.Bounds = bounds;
    pair.Geometry = geometry;


    // check if we can add the object to us first
    bool added = false;
    if ( ( _objects.GetSize() < _countBeforeSubivide ) && ( !HasSubdivided() ) )
    {
        _objects.Add( pair );
        added = true;
    }
    else
    {
        // subdivide if we need to
        if ( !HasSubdivided() )
        {
            Subdivide();
        }

        // try to add the object to children first
        for ( uint32 i = 0; i < 8; ++i )
        {
            if ( _children[ i ]->Add( geometry ) )
            {
                added = true;
                break;
            }
        }

        // now we just add the object to us if no child took it
        if ( !added )
        {
            _objects.Add( pair );
            added = true;
        }
    }

    return added;
}

// subdivides this octree
__device__ void Octree::Subdivide()
{
    // get helper variables
    vec3 center( _bounds.GetCenter() );
    vec3 qdim  ( _bounds.GetSize() * 0.25f );

    // get child centers
    vec3 trb( center.x + qdim.x, center.y + qdim.y, center.z + qdim.z );
    vec3 trf( center.x + qdim.x, center.y + qdim.y, center.z - qdim.z );
    vec3 brb( center.x + qdim.x, center.y - qdim.y, center.z + qdim.z );
    vec3 brf( center.x + qdim.x, center.y - qdim.y, center.z - qdim.z );
    vec3 tlb( center.x - qdim.x, center.y + qdim.y, center.z + qdim.z );
    vec3 tlf( center.x - qdim.x, center.y + qdim.y, center.z - qdim.z );
    vec3 blb( center.x - qdim.x, center.y - qdim.y, center.z + qdim.z );
    vec3 blf( center.x - qdim.x, center.y - qdim.y, center.z - qdim.z );

    // create children
    _children[ 0 ] = new Octree( tlb - qdim, tlb + qdim ); // top left back
    _children[ 1 ] = new Octree( tlf - qdim, tlf + qdim ); // top left front
    _children[ 2 ] = new Octree( trb - qdim, trb + qdim ); // top right back
    _children[ 3 ] = new Octree( trf - qdim, trf + qdim ); // top right front
    _children[ 4 ] = new Octree( blb - qdim, blb + qdim ); // bottom left back
    _children[ 5 ] = new Octree( blf - qdim, blf + qdim ); // bottom left front
    _children[ 6 ] = new Octree( brb - qdim, brb + qdim ); // bottom right back
    _children[ 7 ] = new Octree( brf - qdim, brf + qdim ); // bottom right front

    // go through the new children to see if we can move objects
    for ( size_t oi = 0; oi < _objects.GetSize(); ++oi )
    {
        auto& obj = _objects[ oi ];

        // check each child to see if we can move the object
        for ( uint32 ci = 0; ci < 8; ++ci )
        {
            if ( _children[ ci ]->Add( obj.Geometry ) )
            {
                _objects.Remove( oi );
                --oi;
                break;
            }
        }
    }
}

REX_NS_END