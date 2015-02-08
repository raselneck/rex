#ifndef __REX_SCENE_HXX
#define __REX_SCENE_HXX
#pragma once

#include "Config.hxx"
#include "ViewPlane.hxx"
#include "RayTracer.hxx"
#include "Sphere.hxx"
#include "Image.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a scene.
/// </summary>
class Scene
{
    ViewPlane _viewPlane;
    Color _bgColor;
    Sphere _sphere;
    Handle<RayTracer> _tracer;
    Handle<Image> _image;

public:
    /// <summary>
    /// Creates a new scene.
    /// </summary>
    Scene();

    /// <summary>
    /// Destroys this scene.
    /// </summary>
    ~Scene();

    /// <summary>
    /// Gets the scene's image.
    /// </summary>
    const Handle<Image>& GetImage() const;

    /// <summary>
    /// Gets the scene's sphere.
    /// </summary>
    const Sphere& GetSphere() const;

    /// <summary>
    /// Builds the scene.
    /// </summary>
    /// <param name="hres">The horizontal resolution.</param>
    /// <param name="vres">The vertical resolution.</param>
    void Build( int32 hres, int32 vres );

    /// <summary>
    /// Renders the scene to the image.
    /// </summary>
    void Render();

    /// <summary>
    /// Sets the ray tracer's type.
    /// </summary>
    /// <param name="T">The ray tracer type.</param>
    template<class T> void SetTracerType()
    {
        _tracer.reset( new T( this ) );
    }
};

REX_NS_END

#endif