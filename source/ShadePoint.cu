#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// create new shade point information
ShadePoint::ShadePoint( rex::Scene* scene )
    : Scene( scene )
{
    T           = 0.0;
    Scene       = nullptr;
    Material    = nullptr;
    RecursDepth = 0;
    HasHit      = 0;
}

// destroy shade point
ShadePoint::~ShadePoint()
{
    T           = 0.0;
    Scene       = nullptr;
    Material    = nullptr;
    RecursDepth = 0;
    HasHit      = 0;
}

REX_NS_END