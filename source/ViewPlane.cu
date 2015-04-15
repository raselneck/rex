#include <rex/Graphics/ViewPlane.hxx>

REX_NS_BEGIN

// create a new view plane
ViewPlane::ViewPlane()
{
    Width       = 0;
    Height      = 0;
    SampleCount = 1;
}

// destroy this view plane
ViewPlane::~ViewPlane()
{
    Width       = 0;
    Height      = 0;
    SampleCount = 0;
}

REX_NS_END