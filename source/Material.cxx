#include "Material.hxx"

REX_NS_BEGIN

// create material
Material::Material()
{
}

// destroy material
Material::~Material()
{
}

// area light shade
Color Material::AreaLightShade( ShadePoint& sp )
{
    return Color::Magenta;
}

// path shade
Color Material::PathShade( ShadePoint& sp )
{
    return Color::Magenta;
}

// (ray cast) shade
Color Material::Shade( ShadePoint& sp )
{
    return Color::Magenta;
}

// Whitted shade
Color Material::WhittedShade( ShadePoint& sp )
{
    return Color::Magenta;
}

REX_NS_END