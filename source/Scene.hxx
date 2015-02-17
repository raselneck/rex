#ifndef __REX_SCENE_HXX
#define __REX_SCENE_HXX

#include "Config.hxx"
#include "Camera.hxx"
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
    Handle<Camera>  _camera;
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
    /// Gets this scene's camera.
    /// </summary>
    const Handle<Camera>& GetCamera() const;

    /// <summary>
    /// Gets the scene's image.
    /// </summary>
    const Handle<Image>& GetImage() const;

    /// <summary>
    /// Gets the scene's sampler.
    /// </summary>
    const Handle<Sampler>& GetSampler() const;

    /// <summary>
    /// Gets the scene's tracer.
    /// </summary>
    const Handle<Tracer>& GetTracer() const;

    /// <summary>
    /// Gets this scene's view plane.
    /// </summary>
    const ViewPlane& GetViewPlane() const;

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
    void AddPlane( const Vector3& point, const Vector3& normal );

    /// <summary>
    /// Adds a plane to the scene.
    /// </summary>
    /// <param name="point">A point the plane passes through.</param>
    /// <param name="normal">The plane's normal.</param>
    /// <param name="color">The plane's color.</param>
    void AddPlane( const Vector3& point, const Vector3& normal, const Color& color );

    /// <summary>
    /// Adds a sphere to the scene.
    /// </summary>
    /// <param name="center">The center of the sphere.</param>
    /// <param name="radius">The radius of the sphere.</param>
    void AddSphere( const Vector3& center, real64 radius );

    /// <summary>
    /// Adds a sphere to the scene.
    /// </summary>
    /// <param name="center">The center of the sphere.</param>
    /// <param name="radius">The radius of the sphere.</param>
    /// <param name="color">The plane's color.</param>
    void AddSphere( const Vector3& center, real64 radius, const Color& color );

    /// <summary>
    /// Builds the scene.
    /// </summary>
    /// <param name="hres">The horizontal resolution.</param>
    /// <param name="vres">The vertical resolution.</param>
    /// <param name="ps">The pixel size to use.</param>
    void Build( int32 hres, int32 vres, real32 ps );

    /// <summary>
    /// Gets this scene's camera.
    /// </summary>
    Handle<Camera>& GetCamera();

    /// <summary>
    /// Gets the scene's image.
    /// </summary>
    Handle<Image>& GetImage();

    /// <summary>
    /// Gets the scene's sampler.
    /// </summary>
    Handle<Sampler>& GetSampler();

    /// <summary>
    /// Gets the scene's tracer.
    /// </summary>
    Handle<Tracer>& GetTracer();

    /// <summary>
    /// Renders the scene to the image.
    /// </summary>
    void Render();

    /// <summary>
    /// Sets the scene's camera type.
    /// </summary>
    template<class T> void SetCameraType();

    /// <summary>
    /// Sets the scene's sampler type.
    /// </summary>
    /// <param name="T">The sampler type.</param>
    template<class T> void SetSamplerType();

    /// <summary>
    /// Sets the scene's sampler type.
    /// </summary>
    /// <param name="T">The sampler type.</param>
    /// <param name="samples">The sample count.</param>
    template<class T> void SetSamplerType( int32 samples );

    /// <summary>
    /// Sets the scene's sampler type.
    /// </summary>
    /// <param name="T">The sampler type.</param>
    /// <param name="samples">The sample count.</param>
    /// <param name="sets">The set count.</param>
    template<class T> void SetSamplerType( int32 samples, int32 sets );

    /// <summary>
    /// Sets the scene's ray tracer type.
    /// </summary>
    /// <param name="T">The ray tracer type.</param>
    template<class T> void SetTracerType();
};

REX_NS_END

#include "Scene.inl"
#endif