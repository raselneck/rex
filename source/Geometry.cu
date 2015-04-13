#include <rex/Graphics/Geometry/Geometry.hxx>
#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// destroys this piece of geometry
__device__ Geometry::~Geometry()
{
    if ( _material )
    {
        delete _material;
        _material = nullptr;
    }
}

// get device material
__device__ const Material* Geometry::GetMaterial() const
{
    return _material;
}

// get geometry type
__device__ GeometryType Geometry::GetType() const
{
    return _geometryType;
}

REX_NS_END