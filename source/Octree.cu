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


/// <summary>
/// Creates a new bounding box / geometry pair.
/// </summary>
/// <param name="bounds">The bounding box to use.</param>
/// <param name="geom">The geometry to use.</param>
__device__ static BoundsGeometryPair MakePair( const BoundingBox& bounds, const Geometry* geom )
{
    BoundsGeometryPair pair;
    pair.Bounds = bounds;
    pair.Geometry = geom;
    return pair;
}



// create a new bounding box / geometry pair
__device__ BoundsGeometryPair::BoundsGeometryPair()
    : Bounds( Vector3(), Vector3() )
{
}



// create an octree w/ bounds
__device__ Octree::Octree( const BoundingBox& bounds )
    : Octree( bounds, DEFAULT_MAX_ITEM_COUNT )
{
}

// create an octree w/ min and max corner
__device__ Octree::Octree( const Vector3& min, const Vector3& max )
    : Octree( BoundingBox( min, max ), DEFAULT_MAX_ITEM_COUNT )
{
}

// create an octree w/ bounds and max item count
__device__ Octree::Octree( const BoundingBox& bounds, uint_t maxItemCount )
    : _bounds( bounds ),
      _countBeforeSubivide( maxItemCount )
{
    // clear out the children
    for ( uint_t i = 0; i < 8; ++i )
    {
        _children[ i ] = nullptr;
    }
}

// create an octree w/ min corner, max corner, and max item count
__device__ Octree::Octree( const Vector3& min, const Vector3& max, uint_t maxItemCount )
    : Octree( BoundingBox( min, max ), maxItemCount )
{
}

// destroy this octree
__device__ Octree::~Octree()
{
    if ( HasSubdivided() )
    {
        for ( uint_t i = 0; i < 8; ++i )
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
__device__ const Geometry* Octree::QueryIntersections( const Ray& ray, real_t& dist, ShadePoint& sp ) const
{
    // reset the distance
    dist = Math::HugeValue();


    // make sure the ray even intersects us
    real_t tempDist = 0.0;
    if ( _bounds.Intersects( ray, tempDist ) )
    {
        return QueryIntersectionsForReal( ray, dist, sp );
    }


    // if we don't, then return nothing
    return nullptr;
}

// queries the intersections for real this time
__device__ const Geometry* Octree::QueryIntersectionsForReal( const Ray& ray, real_t& dist, ShadePoint& sp ) const
{
    const Geometry* closest   = nullptr;
    real_t          tempDist  = 0.0;
    ShadePoint      tempPoint = sp;

    // check our children first
    if ( HasSubdivided() )
    {
        for ( uint_t i = 0; i < 8; ++i )
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
    for ( uint_t i = 0; i < _objects.GetSize(); ++i )
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
__device__ bool Octree::QueryShadowRay( const Ray& ray, real_t& dist ) const
{
    real_t d = 0.0;
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
        for ( uint_t i = 0; i < 8; ++i )
        {
            if ( _children[ i ]->QueryShadowRay( ray, d ) && ( d < dist ) )
            {
                hit  = true;
                dist = d;
            }
        }
    }

    // now check all of our objects
    uint_t count = _objects.GetSize();
    for ( uint_t i = 0; i < count; ++i )
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
    BoundingBox bounds = geometry->GetBounds();


    // make sure we contain the geometry's bounding box
    ContainmentType ctype = _bounds.Contains( bounds );
    if ( ctype != ContainmentType::Contains )
    {
        return false;
    }


    // check if we can add the object to us first
    bool added = false;
    if ( ( _objects.GetSize() < _countBeforeSubivide ) && ( !HasSubdivided() ) )
    {
        _objects.Add( MakePair( bounds, geometry ) );

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
        for ( uint_t i = 0; i < 8; ++i )
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
            _objects.Add( MakePair( bounds, geometry ) );

            added = true;
        }
    }

    return added;
}

// subdivides this octree
__device__ void Octree::Subdivide()
{
    // get helper variables
    Vector3 center( _bounds.GetCenter() );
    Vector3 qdim  ( _bounds.GetSize() * 0.25 );

    // get child centers
    Vector3 trb( center.X + qdim.X, center.Y + qdim.Y, center.Z + qdim.Z );
    Vector3 trf( center.X + qdim.X, center.Y + qdim.Y, center.Z - qdim.Z );
    Vector3 brb( center.X + qdim.X, center.Y - qdim.Y, center.Z + qdim.Z );
    Vector3 brf( center.X + qdim.X, center.Y - qdim.Y, center.Z - qdim.Z );
    Vector3 tlb( center.X - qdim.X, center.Y + qdim.Y, center.Z + qdim.Z );
    Vector3 tlf( center.X - qdim.X, center.Y + qdim.Y, center.Z - qdim.Z );
    Vector3 blb( center.X - qdim.X, center.Y - qdim.Y, center.Z + qdim.Z );
    Vector3 blf( center.X - qdim.X, center.Y - qdim.Y, center.Z - qdim.Z );

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
        for ( uint_t ci = 0; ci < 8; ++ci )
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