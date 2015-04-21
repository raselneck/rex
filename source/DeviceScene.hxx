#pragma once

#include <rex/Rex.hxx>

REX_NS_BEGIN

/// <summary>
/// Contains scene data destined for a device.
/// </summary>
struct DeviceSceneData
{
    const DeviceList<Light*>* Lights;
    const AmbientLight*       AmbientLight;
    const Octree*             Octree;
    const Camera              Camera;
    const ViewPlane           ViewPlane;
    const Color               BackgroundColor;
    uchar4*                   Pixels;
};

/// <summary>
/// The scene render kernel.
/// </summary>
/// <param name="sd">The scene data.</param>
__global__ void SceneRenderKernel( DeviceSceneData* sd );

REX_NS_END