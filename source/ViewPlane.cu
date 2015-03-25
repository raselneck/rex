#include <rex/Graphics/ViewPlane.hxx>

REX_NS_BEGIN

// create a new view plane
ViewPlane::ViewPlane()
{
    Width       = 0;
    Height      = 0;
    SampleCount = 1;
    PixelSize   = 1.0f;
    Gamma       = 2.2f;
    InvGamma    = 1.0f / Gamma;
}

// destroy this view plane
ViewPlane::~ViewPlane()
{
    Width       = 0;
    Height      = 0;
    SampleCount = 0;
    PixelSize   = 0;
    Gamma       = 0;
    InvGamma    = 0;
}

REX_NS_END