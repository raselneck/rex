#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// create new shade point information
ShadePoint::ShadePoint()
{
    T           = 0.0;
    Material    = nullptr;
    HasHit      = 0;
}

// destroy shade point
ShadePoint::~ShadePoint()
{
    T           = 0.0;
    Material    = nullptr;
    HasHit      = 0;
}

REX_NS_END