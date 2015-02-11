#ifndef __REX_SCENE_HXX
#define __REX_SCENE_HXX
#pragma once

#include "Config.hxx"
#include "Geometry.hxx"
#include "Image.hxx"
#include "Tracer.hxx"
#include "Sampler.hxx"
#include "ShadePoint.hxx"
#include "ViewPlane.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines a scene.
/// </summary>
class Scene
{
    ViewPlane       _viewPlane;
    Color           _bgColor;
    Handle<Image>   _image;
    Handle<Tracer>  _tracer;
    Handle<Sampler> _sampler;
    std::vector<Handle<Geometry>> _objects;

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
    /// Gets this scene's background color.
    /// </summary>
    const Color& GetBackgroundColor() const;

    /// <summary>
    /// Gets the scene's image.
    /// </summary>
    const Handle<Image>& GetImage() const;

    /// <summary>
    /// Hits all of the objects in this scene with the given ray.
    /// </summary>
    /// <param name="ray">The ray to hit with.</param>
    ShadePoint HitObjects( const Ray& ray ) const;

    /// <summary>
    /// Adds a plane to the scene.
    /// </summary>
    /// <param name="point">A point the plane passes through.</param>
    /// <param name="normal">The plane's normal.</param>
    /// <param name="color">The plane's color.</param>
    void AddPlane( const Vector3& point, const Vector3& normal, const Color& color = Color::Black );

    /// <summary>
    /// Adds a sphere to the scene.
    /// </summary>
    /// <param name="center">The center of the sphere.</param>
    /// <param name="radius">The radius of the sphere.</param>
    /// <param name="color">The plane's color.</param>
    void AddSphere( const Vector3& center, real64 radius, const Color& color = Color::Black );

    /// <summary>
    /// Builds the scene.
    /// </summary>
    /// <param name="hres">The horizontal resolution.</param>
    /// <param name="vres">The vertical resolution.</param>
    /// <param name="ps">The pixel size to use.</param>
    void Build( int32 hres, int32 vres, real32 ps );

    /// <summary>
    /// Renders the scene to the image.
    /// </summary>
    void Render();

    /// <summary>
    /// Sets the scene's sampler type.
    /// </summary>
    /// <param name="T">The sampler type.</param>
    /// <param name="samples">The sample count.</param>
    /// <param name="sets">The set count.</param>
    template<class T> void SetSamplerType( int32 samples, int32 sets )
    {
        _sampler.reset( new T( samples, sets ) );
        _sampler->GenerateSamples();
    }

    /// <summary>
    /// Sets the scene's ray tracer type.
    /// </summary>
    /// <param name="T">The ray tracer type.</param>
    template<class T> void SetTracerType()
    {
        _tracer.reset( new T( this ) );
    }
};

REX_NS_END

#endif