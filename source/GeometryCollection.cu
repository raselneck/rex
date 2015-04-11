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
{
}

// destroys this geometry collection
GeometryCollection::~GeometryCollection()
{
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

// get device geometry
const std::vector<const Geometry*>& GeometryCollection::GetDeviceGeometry() const
{
    return _dGeometry;
}

// add a sphere
Sphere* GeometryCollection::AddSphere()
{
    auto sphere = GC::HostAlloc<Sphere>( defaultGeometryMaterial );
    if ( sphere )
    {
        _hGeometry.push_back( sphere );
        _dGeometry.push_back( sphere->GetOnDevice() );
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
    }
    return sphere;
}

REX_NS_END