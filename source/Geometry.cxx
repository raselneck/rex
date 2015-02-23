#include <rex/Geometry/Geometry.hxx>
#include <rex/Materials/MatteMaterial.hxx>

REX_NS_BEGIN

// gets the default material for geometric objects
static inline MatteMaterial GetDefaultMaterial()
{
    MatteMaterial matte;
    matte.SetAmbientCoefficient( 0.25f );
    matte.SetDiffuseCoefficient( 0.75f );
    return matte;
}

// new geometry object
Geometry::Geometry()
{
    SetMaterial( GetDefaultMaterial() );
}

// destroy geometry object
Geometry::~Geometry()
{
}

// get geometry material
const Material* Geometry::GetMaterial() const
{
    return _material.get();
}

REX_NS_END