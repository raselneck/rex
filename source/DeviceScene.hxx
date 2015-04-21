#pragma once

#include <rex/Rex.hxx>

REX_NS_BEGIN


// TODO : Make a unified memory source so that the kernel only has to render to that. To do
// that I just need to change the image's device memory to be a uchar3 pointer.


/// <summary>
/// Contains scene data destined for a device.
/// </summary>
struct DeviceSceneData
{
    const DeviceList<Light*>* Lights;
    const AmbientLight*       AmbientLight;
    const Camera*             Camera;
    const Octree*             Octree;
    const ViewPlane*          ViewPlane;
    const Color*              BackgroundColor;
    Image*                    Image;
    uchar3*                   TextureMemory;
};

/// <summary>
/// The scene render kernel.
/// </summary>
/// <param name="sd">The scene data.</param>
__global__ void SceneRenderKernel( DeviceSceneData* sd );

REX_NS_END