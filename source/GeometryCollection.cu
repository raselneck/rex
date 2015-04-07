#include <rex/Graphics/Geometry/GeometryCollection.hxx>
#include <rex/Graphics/Materials/MatteMaterial.hxx>
#include <rex/Utility/GC.hxx>
#include <rex/Utility/Logger.hxx>

REX_NS_BEGIN

/// <summary>
/// The default material used for geometry in a geometry collection.
/// </summary>
static MatteMaterial defaultGeometryMaterial = MatteMaterial( Color::White(), 1.0f, 1.0f );

// creates a new geometry collection
GeometryCollection::GeometryCollection()
    : _dGeometryArray( nullptr )
{
    // initialize the device array
    UpdateDeviceArray();
}

// destroys this geometry collection
GeometryCollection::~GeometryCollection()
{
    if ( _dGeometryArray )
    {
        cudaFree( _dGeometryArray );
        _dGeometryArray = nullptr;
    }
}

// get number of geometric objects
uint32 GeometryCollection::GetGeometryCount() const
{
    return static_cast<uint32>( _hGeometry.size() );
}

// get the host geometry
const std::vector<Geometry*>& GeometryCollection::GetGeometry() const
{
    return _hGeometry;
}

// get the device geometry
const Geometry** GeometryCollection::GetDeviceGeometry() const
{
    return (const Geometry**)( _dGeometryArray );
}

// add a sphere
Sphere* GeometryCollection::AddSphere()
{
    auto sphere = GC::HostAlloc<Sphere>( defaultGeometryMaterial );
    if ( sphere )
    {
        _hGeometry.push_back( sphere );
        _dGeometry.push_back( sphere->GetOnDevice() );

        UpdateDeviceArray();
    }
    return sphere;
}

// add a sphere
Sphere* GeometryCollection::AddSphere( const Vector3& center, real64 radius )
{
    auto sphere = GC::HostAlloc<Sphere>( defaultGeometryMaterial, center, radius );
    if ( sphere )
    {
        _hGeometry.push_back( sphere );
        _dGeometry.push_back( sphere->GetOnDevice() );

        UpdateDeviceArray();
    }
    return sphere;
}

// update the device array
void GeometryCollection::UpdateDeviceArray()
{
    // cleanup the old array
    if ( _dGeometryArray )
    {
        cudaFree( _dGeometryArray );
        _dGeometryArray = nullptr;
    }


    // create the array
    const uint32 byteCount = _hGeometry.size() * sizeof( Geometry* );
    cudaError_t  err       = cudaMalloc( reinterpret_cast<void**>( &_dGeometryArray ), byteCount );
    if ( err != cudaSuccess )
    {
        Logger::Log( "Failed to allocate space for geometry collection on device." );
        return;
    }


    // copy over the device pointers
    if ( _dGeometry.size() > 0 )
    {
        err = cudaMemcpy( _dGeometryArray, &( _dGeometry[ 0 ] ), byteCount, cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            Logger::Log( "Allocated geometry collection on device, but failed to copy data." );
            cudaFree( _dGeometryArray );
            _dGeometryArray = nullptr;
        }
    }
}

REX_NS_END