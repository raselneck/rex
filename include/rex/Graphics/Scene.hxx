#pragma once

#include "../Config.hxx"
#include "Lights/AmbientLight.hxx"
#include "Lights/DirectionalLight.hxx"
#include "Lights/PointLight.hxx"
#include "Camera.hxx"
#include "ShadePoint.hxx"
#include "ViewPlane.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a scene.
/// </summary>
class Scene
{
    ViewPlane _viewPlane;
    Color     _backgroundColor;
    Camera    _camera;

    // prevent copying and moving of scenes
    Scene( const Scene& ) = delete;
    Scene( Scene&& ) = delete;
    Scene& operator=( const Scene& ) = delete;
    Scene& operator=( Scene&& ) = delete;

public:
    /// <summary>
    /// Creates a new scene.
    /// </summary>
    Scene();

    /// <summary>
    /// Destroys this scene.
    /// </summary>
    ~Scene();
};

REX_NS_END