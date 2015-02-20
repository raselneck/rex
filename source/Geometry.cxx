#include "Geometry.hxx"

REX_NS_BEGIN

// new geometry object
Geometry::Geometry()
{
}

// new geometry object
Geometry::Geometry( const rex::Color& color )
    : _color( color )
{
}

// destroy geometry object
Geometry::~Geometry()
{
}

// get geometry color
const Color& Geometry::GetColor() const
{
    return _color;
}

REX_NS_END