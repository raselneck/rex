#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// create new shade point information
ShadePoint::ShadePoint()
{
    T           = 0.0;
    Material    = nullptr;
}

// destroy shade point
ShadePoint::~ShadePoint()
{
    T           = 0.0;
    Material    = nullptr;
}

REX_NS_END