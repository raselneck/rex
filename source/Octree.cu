#include <rex/Graphics/Geometry/Octree.hxx>
#include <rex/Graphics/Geometry/Geometry.hxx>
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
__host__ static BoundsGeometryPair MakePair( const BoundingBox& bounds, const Geometry* geom )
{
    BoundsGeometryPair pair;
    pair.Bounds = bounds;
    pair.Geometry = geom;
    return pair;
}


// create a new bounding box / geometry pair
BoundsGeometryPair::BoundsGeometryPair()
    : Bounds( Vector3(), Vector3() )
{
}



// create a new device octree
DeviceOctree::DeviceOctree()
    : Bounds( Vector3(), Vector3() )
{
    // set the children to null
    for ( uint32 i = 0; i < 8; ++i )
    {
        Children[ i ] = nullptr;
    }

    // register that we have no objects
    Objects     = nullptr;
    ObjectCount = 0;
}

// query the intersections of the given ray and return the closest hit object
__device__ const Geometry* DeviceOctree::QueryIntersections( const Ray& ray, real64& dist ) const
{
    // reset the distance
    dist = Math::HugeValue();


    // make sure the ray even intersects us
    real64 tempDist = 0.0;
    if ( Bounds.Intersects( ray, tempDist ) )
    {
        return QueryIntersectionsForReal( ray, dist );
    }


    // if we don't, then return nothing
    return nullptr;
}

// queries the intersections for real this time
__device__ const Geometry* DeviceOctree::QueryIntersectionsForReal( const Ray& ray, real64& dist ) const
{
    const Geometry* closest = nullptr;
    real64          tempDist = 0.0;

    // check our children first
    if ( Children[ 0 ] != nullptr )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            // have the child query for ray intersections
            const DeviceOctree* child = Children[ i ];
            const Geometry*     geom = child->QueryIntersections( ray, tempDist );

            // check to see if the child intersected the ray
            if ( ( geom != nullptr ) && ( tempDist < dist ) )
            {
                closest = geom;
                dist = tempDist;
            }
        }
    }

    // now check our objects
    for ( uint32 i = 0; i < ObjectCount; ++i )
    {
        BoundsGeometryPair& pair = Objects[ i ];
        if ( pair.Bounds.Intersects( ray, tempDist ) && ( tempDist < dist ) )
        {
            closest = pair.Geometry;
            dist    = tempDist;
        }
    }

    return closest;
}

// queries the intersections of the given ray for shadows
__device__ bool DeviceOctree::QueryShadowRay( const Ray& ray ) const
{
    // ensure the ray even intersects our bounds
    real64 dist = 0.0;
    if ( !Bounds.Intersects( ray, dist ) )
    {
        return false;
    }

    // check the children first if we have subdivided
    if ( Children[ 0 ] != nullptr )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            if ( Children[ i ]->QueryShadowRay( ray ) )
            {
                return true;
            }
        }
    }

    // now check all of our objects
    for ( uint32 i = 0; i < 8; ++i )
    {
        if ( Objects[ i ].Geometry->ShadowHit( ray, dist ) )
        {
            return true;
        }
    }

    return false;
}




// create an octree w/ bounds
Octree::Octree( const BoundingBox& bounds )
    : Octree( bounds.GetMin(), bounds.GetMax(), DEFAULT_MAX_ITEM_COUNT )
{
}

// create an octree w/ min and max corner
Octree::Octree( const Vector3& min, const Vector3& max )
    : Octree( min, max, DEFAULT_MAX_ITEM_COUNT )
{
}

// create an octree w/ min corner, max corner, and max item count
Octree::Octree( const Vector3& min, const Vector3& max, uint32 maxItemCount )
    : _bounds( min, max ),
      _countBeforeSubivide( maxItemCount ),
      _isDevicePointerStale( true ),
      _dThis( nullptr ),
      _dThisObjects( nullptr )
{
    // clear out the children
    for ( uint32 i = 0; i < 8; ++i )
    {
        _hChildren[ i ] = nullptr;
    }


    // allocate our "this" pointer
    DeviceOctree dThis;
    _dThis = GC::DeviceAlloc<DeviceOctree>( dThis );


    // create the initial device array
    UpdateDeviceArray();
}

// destroy this octree
Octree::~Octree()
{
    // we need to free the object list before the device pointer itself (handled by GC)
    if ( _dThisObjects )
    {
        cudaFree( _dThisObjects );
        _dThisObjects = nullptr;
    }
}

// get the octree's bounds
const BoundingBox& Octree::GetBounds() const
{
    return _bounds;
}

// check if this octree has subdivided
bool Octree::HasSubdivided() const
{
    return _hChildren[ 0 ] != nullptr;
}

// get this octree on the device
const DeviceOctree* Octree::GetOnDevice()
{
    if ( _isDevicePointerStale )
    {
        UpdateDeviceArray();
    }

    return (const DeviceOctree*)( _dThis );
}

// add the given piece of geometry to this octree
bool Octree::Add( const Geometry* geometry )
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
    if ( ( _hObjects.size() < _countBeforeSubivide ) && ( !HasSubdivided() ) )
    {
        _hObjects.push_back( MakePair( bounds, geometry ) );
        _dObjects.push_back( MakePair( bounds, geometry->GetOnDevice() ) );

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
            if ( _hChildren[ i ]->Add( geometry ) )
            {
                added = true;
                break;
            }
        }

        // now we just add the object to us if no child took it
        if ( !added )
        {
            _hObjects.push_back( MakePair( bounds, geometry ) );
            _dObjects.push_back( MakePair( bounds, geometry->GetOnDevice() ) );

            added = true;
        }
    }


    // update our cache flag and return whether or not the piece of geometry was added
    _isDevicePointerStale = true;
    return added;
}

// subdivides this octree
void Octree::Subdivide()
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
    _hChildren[ 0 ] = GC::HostAlloc<Octree>( tlb - qdim, tlb + qdim ); // top left back
    _hChildren[ 1 ] = GC::HostAlloc<Octree>( tlf - qdim, tlf + qdim ); // top left front
    _hChildren[ 2 ] = GC::HostAlloc<Octree>( trb - qdim, trb + qdim ); // top right back
    _hChildren[ 3 ] = GC::HostAlloc<Octree>( trf - qdim, trf + qdim ); // top right front
    _hChildren[ 4 ] = GC::HostAlloc<Octree>( blb - qdim, blb + qdim ); // bottom left back
    _hChildren[ 5 ] = GC::HostAlloc<Octree>( blf - qdim, blf + qdim ); // bottom left front
    _hChildren[ 6 ] = GC::HostAlloc<Octree>( brb - qdim, brb + qdim ); // bottom right back
    _hChildren[ 7 ] = GC::HostAlloc<Octree>( brf - qdim, brf + qdim ); // bottom right front

    // go through the new children to see if we can move objects
    for ( size_t oi = 0; oi < _hObjects.size(); ++oi )
    {
        auto& obj = _hObjects[ oi ];

        // check each child to see if we can move the object
        for ( uint32 ci = 0; ci < 8; ++ci )
        {
            if ( _hChildren[ ci ]->Add( obj.Geometry ) )
            {
                _hObjects.erase( _hObjects.begin() + oi );
                _dObjects.erase( _dObjects.begin() + oi );
                --oi;
                break;
            }
        }
    }

    _isDevicePointerStale = true;
}

// update the device array
void Octree::UpdateDeviceArray()
{
    // if we're not stale, don't unnecessarily update the array
    if ( !_isDevicePointerStale )
    {
        return;
    }



    // free our old memory if necessary
    if ( _dThisObjects )
    {
        GC::UnregisterDeviceMemory( _dThisObjects );
        cudaFree( _dThisObjects );
        _dThisObjects = nullptr;
    }



    // first tell our children to update their arrays
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            _hChildren[ i ]->UpdateDeviceArray();
        }
    }



    // setup our device pointer
    DeviceOctree dThis;
    dThis.Bounds = _bounds;
    dThis.ObjectCount = _dObjects.size();
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            dThis.Children[ i ] = _hChildren[ i ]->_dThis;
        }
    }

    // allocate space for the objects on the device and copy over the data
    cudaError_t err;
    if ( _dObjects.size() > 0 )
    {
        dThis.Objects = GC::DeviceAllocArray<BoundsGeometryPair>( _dObjects.size(), &( _dObjects[ 0 ] ) );
        _dThisObjects = dThis.Objects;
    }



    // copy over us to the device
    _isDevicePointerStale = false;
    err = cudaMemcpy( _dThis, &dThis, sizeof( DeviceOctree ), cudaMemcpyHostToDevice );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to copy device octree's objects." );
        _isDevicePointerStale = true;
        return;
    }
}

REX_NS_END