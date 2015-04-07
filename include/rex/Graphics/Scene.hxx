#pragma once

#include "../Config.hxx"
#include "../Utility/Image.hxx"
#include "Geometry/Octree.hxx"
#include "Geometry/GeometryCollection.hxx"
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

    ViewPlane           _viewPlane;
    Color               _backgroundColor;
    Camera              _camera;
    Handle<Image>       _image;
    LightCollection     _lights;
    GeometryCollection  _geometry;
    Handle<Octree>      _octree;

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
    __host__ bool Build( uint16 width, uint16 height );

    /// <summary>
    /// Renders this scene.
    /// </summary>
    __host__ void Render();
};

REX_NS_END