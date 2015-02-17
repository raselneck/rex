#ifndef __REX_VIEWPLANE_HXX
#define __REX_VIEWPLANE_HXX

#include "Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a view plane.
/// </summary>
struct ViewPlane
{
    int32 Width;
    int32 Height;
    real32 PixelSize;
    real32 Gamma;
    real32 InvGamma;

    /// <summary>
    /// Creates a new view plane.
    /// </summary>
    ViewPlane();

    /// <summary>
    /// Destroys this view plane.
    /// </summary>
    ~ViewPlane();
};

REX_NS_END

#endif