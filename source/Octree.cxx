#include <rex/Geometry/Octree.hxx>
#include <rex/Geometry/Geometry.hxx>

REX_NS_BEGIN

// the maximum number of children an octree can have before it subdivides
static const uint32 MaxOctreeChildCount = 8;


// create octree
Octree::Octree( const BoundingBox& bounds )
    : _bounds( bounds )
{
    for ( uint32 i = 0; i < 8; ++i )
    {
        _children[ i ] = nullptr;
    }
}

// create octree
Octree::Octree( const Vector3& min, const Vector3& max )
    : _bounds( min, max )
{
    for ( uint32 i = 0; i < 8; ++i )
    {
        _children[ i ] = nullptr;
    }
}

// destroy octree
Octree::~Octree()
{
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            delete _children[ i ];
        }
    }
}

// check if we've subdivided
bool Octree::HasSubdivided() const
{
    return _children[ 0 ] != nullptr;
}

// query intersections
void Octree::QueryIntersections( const Ray& ray, std::vector<const Geometry*>& objects ) const
{
    objects.clear();

    real64 dist = 0.0;
    if ( _bounds.Intersects( ray, dist ) )
    {
        QueryIntersectionsRecurs( ray, objects );
    }
}

// query intersections
void Octree::QueryIntersectionsRecurs( const Ray& ray, std::vector<const Geometry*>& objects ) const
{
    real64 dist = 0.0;

    // check children first
    if ( HasSubdivided() )
    {
        for ( uint32 i = 0; i < 8; ++i )
        {
            auto& child = _children[ i ];
            if ( child->_bounds.Intersects( ray, dist ) )
            {
                _children[ i ]->QueryIntersectionsRecurs( ray, objects );
            }
        }
    }

    // now check our objects
    for ( auto& pair : _objects )
    {
        if ( pair.first.Intersects( ray, dist ) )
        {
            objects.push_back( pair.second );
        }
    }
}

// add geometry to octree
bool Octree::Add( const Geometry* geometry )
{
    BoundingBox gBounds = geometry->GetBounds();

    // make sure we contain the geometry's bounding box
    ContainmentType ctype = _bounds.Contains( gBounds );
    if ( ctype != ContainmentType::Contains )
    {
        return false;
    }

    // check if we can add the object to us first
    if ( _objects.size() < MaxOctreeChildCount && !HasSubdivided() )
    {
        _objects.push_back( std::make_pair( gBounds, geometry ) );
    }
    else
    {
        // check if we need to subdivide
        if ( !HasSubdivided() )
        {
            Subdivide();
        }

        // try to add the object to a child
        bool added = false;
        for ( uint32 i = 0; i < 8; ++i )
        {
            if ( _children[ i ]->Add( geometry ) )
            {
                added = true;
                break;
            }
        }

        // just add the object 
        if ( !added )
        {
            _objects.push_back( std::make_pair( gBounds, geometry ) );
        }
    }

    return true;
}

// subdivide octree
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
    _children[ 0 ] = new Octree( tlb - qdim, tlb + qdim ); // top left back
    _children[ 1 ] = new Octree( tlf - qdim, tlf + qdim ); // top left front
    _children[ 2 ] = new Octree( trb - qdim, trb + qdim ); // top right back
    _children[ 3 ] = new Octree( trf - qdim, trf + qdim ); // top right front
    _children[ 4 ] = new Octree( blb - qdim, blb + qdim ); // bottom left back
    _children[ 5 ] = new Octree( blf - qdim, blf + qdim ); // bottom left front
    _children[ 6 ] = new Octree( brb - qdim, brb + qdim ); // bottom right back
    _children[ 7 ] = new Octree( brf - qdim, brf + qdim ); // bottom right front

    // go through the new children to see if we can move objects
    for ( size_t oi = 0; oi < _objects.size(); ++oi )
    {
        auto& obj = _objects[ oi ];
        
        // check each child to see if we can move the object
        for ( uint32 ci = 0; ci < 8; ++ci )
        {
            if ( _children[ ci ]->Add( obj.second ) )
            {
                _objects.erase( _objects.begin() + oi );
                --oi;
                break;
            }
        }
    }
}

REX_NS_END