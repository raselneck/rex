#include <rex/Graphics/Materials/Material.hxx>
#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// create material
Material::Material()
    : _dThis( nullptr )
{
}

// destroy material
Material::~Material()
{
}

// (ray cast) shade
__device__ Color Material::Shade( ShadePoint& sp, const Light** lights, uint32 lightCount ) const
{
    return Color::Magenta();
}

REX_NS_END