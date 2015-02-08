#include "ViewPlane.hxx"

REX_NS_BEGIN

// new view plane
ViewPlane::ViewPlane()
{
    memset( this, 0, sizeof( ViewPlane ) );

    PixelSize = 1.0f;
    Gamma     = 2.2f;
    InvGamma  = 1.0f / Gamma;
}

// destroy view plane
ViewPlane::~ViewPlane()
{
    memset( this, 0, sizeof( ViewPlane ) );
}

REX_NS_END