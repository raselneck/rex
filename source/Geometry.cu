#include <rex/Graphics/Geometry/Geometry.hxx>

REX_NS_BEGIN

// destroys this piece of geometry
Geometry::~Geometry()
{
    if ( _hMaterial )
    {
        delete _hMaterial;
        _hMaterial = nullptr;
    }

    _dMaterial = nullptr;
}

// default callback for changing the material
void Geometry::OnChangeMaterial()
{
}

REX_NS_END