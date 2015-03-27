#pragma once

#include "../Config.hxx"
#include "../Utility/Image.hxx"
#include "Lights/LightCollection.hxx"
#include "Camera.hxx"
#include "ShadePoint.hxx"
#include "ViewPlane.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a scene.
/// </summary>
class Scene
{
    REX_NONCOPYABLE_CLASS( Scene );

    ViewPlane*      _viewPlane;
    Color*          _backgroundColor;
    Camera*         _camera;
    Image*          _image;
    LightCollection _lights;
    
public:
    /// <summary>
    /// Creates a new scene.
    /// </summary>
    __host__ Scene();

    /// <summary>
    /// Destroys this scene.
    /// </summary>
    __host__ ~Scene();

    /// <summary>
    /// Builds this scene.
    /// </summary>
    /// <param name="width">The width of the rendered scene. The maximum width is 2048.</param>
    /// <param name="height">The height of the rendered scene. The maximum height is 2048.</param>
    __host__ void Build( uint16 width, uint16 height );

    /// <summary>
    /// Renders this scene.
    /// </summary>
    __host__ void Render();

    /// <summary>
    /// Hits all of the objects in this scene with the given ray.
    /// </summary>
    /// <param name="ray">The ray to hit with.</param>
    /// <param name="sp">The shade point object to fill data with.</param>
    __device__ void HitObjects( const Ray& ray, ShadePoint& sp ) const;

    /// <summary>
    /// Shadow-hits all of the objects in this scene with the given ray.
    /// </summary>
    /// <param name="ray">The ray to hit with.</param>
    __device__ bool ShadowHitObjects( const Ray& ray ) const;
};

REX_NS_END