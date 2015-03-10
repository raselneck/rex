#include <rex/Geometry/Mesh.hxx>
#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

/// <summary>
/// Defines a primitive triangle.
/// </summary>
struct PrimitiveTriangle
{
    Vector3 P1;
    Vector3 P2;
    Vector3 P3;
};



// create mesh
Mesh::Mesh()
{
}

// destroy mesh
Mesh::~Mesh()
{
}

// get mesh geometry type
GeometryType Mesh::GetType() const
{
    return GeometryType::Mesh;
}

// get mesh bounds
BoundingBox Mesh::GetBounds() const
{
    if ( _octree )
    {
        return _octree->GetBounds();
    }
    return BoundingBox( Vector3( 0.0 ), Vector3( 0.0 ) );
}

// hit mesh
bool Mesh::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // TODO : Query octree

    return false;
}

// shadow hit mesh
bool Mesh::ShadowHit( const Ray& ray, real64& tmin ) const
{
    // TODO : Query octree

    return false;
}

// load mesh
bool Mesh::Load( const String& fname )
{
    // TODO : Follow along with the video tutorial to implement this
    // https://www.youtube.com/watch?v=ClqnhYAYtcY&list=PLC2D979DC6CF73B47&index=6

    return false;
}

REX_NS_END