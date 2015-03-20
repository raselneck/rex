#include <rex/Graphics/Materials/Material.hxx>

REX_NS_BEGIN

// create material
Material::Material()
{
}

// destroy material
Material::~Material()
{
}

// (ray cast) shade
Color Material::Shade( ShadePoint& sp )
{
    return Color::Magenta();
}

REX_NS_END