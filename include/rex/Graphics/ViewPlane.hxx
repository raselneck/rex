#pragma once

#include "../Config.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a view plane.
/// </summary>
struct ViewPlane
{
    uint32 Width;
    uint32 Height;
    uint32 SampleCount;
    real32 PixelSize;
    real32 Gamma;
    real32 InvGamma;

    /// <summary>
    /// Creates a new view plane.
    /// </summary>
    __both__ ViewPlane();

    /// <summary>
    /// Destroys this view plane.
    /// </summary>
    __both__ ~ViewPlane();
};

REX_NS_END