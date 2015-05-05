#include <rex/Graphics/Materials/Material.hxx>
#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// create material
__device__ Material::Material( MaterialType type )
    : _type( type )
{
}

// destroy material
__device__ Material::~Material()
{
}

// area light shade is an ugly color
__device__ Color Material::AreaLightShade( ShadePoint& sp ) const
{
    return Color::Magenta();
}

// get material type
__device__ MaterialType Material::GetType() const
{
    return _type;
}

// default shade is an ugly color, too
__device__ Color Material::Shade( ShadePoint& sp ) const
{
    return Color::Magenta();
}

REX_NS_END