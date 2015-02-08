#include "Geometry.hxx"

REX_NS_BEGIN

// new geometry object
Geometry::Geometry()
{
}

// new geometry object
Geometry::Geometry( const rex::Color& color )
    : Color( color )
{
}

// destroy geometry object
Geometry::~Geometry()
{
}

REX_NS_END