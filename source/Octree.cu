#include <rex/Graphics/Geometry/Octree.hxx>
#include <rex/Graphics/Geometry/Geometry.hxx>
#include <rex/Math/Math.hxx>
#include <rex/Utility/GC.hxx>
#include <rex/Utility/Logger.hxx>


#define DEFAULT_MAX_ITEM_COUNT 12


REX_NS_BEGIN

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
      _dObjectArray( nullptr ),
      _dObjectCount( 0 ),
      _dThis( nullptr )
{
    // clear out the children
    for ( uint32 i = 0; i < 8; ++i )
    {
        _hChildren[ i ] = nullptr;
        _dChildren[ i ] = nullptr;
    }


    // allocate our "this" pointer
    _dThis = GC::DeviceAlloc<Octree>( *this );


    // create the initial device array
    UpdateDeviceArray();
}

// destroy this octree
Octree::~Octree()
{
    // TODO : Do we need anything here?
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
const Octree* Octree::GetOnDevice()
{
    if ( _isDevicePointerStale )
    {
        UpdateDeviceArray();
    }

    return (const Octree*)( _dThis );
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
        _hObjects.push_back( std::make_pair( bounds, geometry ) );
        _dObjects.push_back( std::make_pair( bounds, geometry->GetOnDevice() ) );

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
            _hObjects.push_back( std::make_pair( bounds, geometry ) );
            _dObjects.push_back( std::make_pair( bounds, geometry->GetOnDevice() ) );

            added = true;
        }
    }


    // update our cache flag and return whether or not the piece of geometry was added
    _isDevicePointerStale = true;
    return added;
}

// query the intersections of the given ray and return the closest hit object
__device__ const Geometry* Octree::QueryIntersections( const Ray& ray, real64& dist ) const
{
    // reset the distance
    dist = Math::HugeValue();

    // make sure the ray even intersects us
    real64 tempDist = 0.0;
    if ( _bounds.Intersects( ray, tempDist ) )
    {
        return QueryIntersectionsForReal( ray, dist );
    }

    // if we don't, then return nothing
    return nullptr;
}

// queries the intersections for real this time
__device__ const Geometry* Octree::QueryIntersectionsForReal( const Ray& ray, real64& dist ) const
{
    const Geometry* closest  = nullptr;
    real64          tempDist = 0.0;

    // check our children first
    if ( _dChildren[ 0 ] != nullptr )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            // have the child query for ray intersections
            Octree* child = _dChildren[ i ];
            const Geometry* geom = child->QueryIntersections( ray, tempDist );

            // check to see if the child intersected the ray
            if ( ( geom != nullptr ) && ( tempDist < dist ) )
            {
                closest = geom;
                dist    = tempDist;
            }
        }
    }

    // now check our objects
    for ( uint32 i = 0; i < _dObjectCount; ++i )
    {
        BoundsGeometryPair* pair = _dObjectArray[ i ];
        if ( pair->first.Intersects( ray, tempDist ) && ( tempDist < dist ) )
        {
            closest = pair->second;
            dist    = tempDist;
        }
    }

    return closest;
}

// queries the intersections of the given ray for shadows
__device__ bool Octree::QueryShadowRay( const Ray& ray ) const
{
    // ensure the ray even intersects our bounds
    real64 dist = 0.0;
    if ( !_bounds.Intersects( ray, dist ) )
    {
        return false;
    }

    // check the children first if we have subdivided
    if ( _dChildren[ 0 ] != nullptr )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            if ( _dChildren[ i ]->QueryShadowRay( ray ) )
            {
                return true;
            }
        }
    }

    // now check all of our objects
    for ( uint32 i = 0; i < 8; ++i )
    {
        if ( _dObjectArray[ i ]->second->ShadowHit( ray, dist ) )
        {
            return true;
        }
    }

    return false;
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
            if ( _hChildren[ ci ]->Add( obj.second ) )
            {
                _hObjects.erase( _hObjects.begin() + oi );
                _dObjects.erase( _dObjects.begin() + oi );
                --oi;
                break;
            }
        }
    }

    // get our children pointers
    for ( uint32 i = 0; i < 8; ++i )
    {
        _dChildren[ i ] = reinterpret_cast<Octree*>( _hChildren[ i ]->_dThis );
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


    // cleanup the old array
    if ( _dObjectArray )
    {
        cudaFree( _dObjectArray );
        _dObjectArray = nullptr;
    }


    // set the object count for when we copy over
    _dObjectCount = static_cast<uint32>( _hObjects.size() );



    // create the array
    const uint32 byteCount = _hObjects.size() * sizeof( BoundsGeometryPair );
    cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &_dObjectArray ), byteCount );
    if ( err != cudaSuccess )
    {
        Logger::Log( "Failed to allocate space for the octree's object array on device." );
        return;
    }


    // now copy over all of the information to the device
    if ( _dObjects.size() > 0 )
    {
        err = cudaMemcpy( _dObjectArray, &( _dObjects[ 0 ] ), byteCount, cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            Logger::Log( "Allocated octree object collection on device, but failed to copy data." );
            cudaFree( _dObjectArray );
            _dObjectArray = nullptr;
            return;
        }
    }


    // now tell all of our children to update their device arrays
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            _hChildren[ i ]->UpdateDeviceArray();
        }
    }



    // update our "this" pointer
    err = cudaMemcpy( _dThis, this, sizeof( Octree ), cudaMemcpyHostToDevice );
    if ( err != cudaSuccess )
    {
        Logger::Log( "Failed to update octree self pointer on device." );
        cudaFree( _dObjectArray );
        _dObjectArray = nullptr;
        return;
    }


    // reset our cache flag
    _isDevicePointerStale = false;
}

REX_NS_END