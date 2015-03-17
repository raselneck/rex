#ifndef __REX_SCENE_HXX
#define __REX_SCENE_HXX

#include "../Config.hxx"
#include "../Cameras/Camera.hxx"
#include "../Geometry/Octree.hxx"
#include "../Geometry/Plane.hxx"
#include "../Geometry/Sphere.hxx"
#include "../Lights/AmbientLight.hxx"
#include "../Lights/DirectionalLight.hxx"
#include "../Lights/PointLight.hxx"
#include "../Samplers/Sampler.hxx"
#include "../Tracers/Tracer.hxx"
#include "../Utility/Image.hxx"
#include "ShadePoint.hxx"
#include "ViewPlane.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines a scene.
/// </summary>
class Scene
{
    ViewPlane            _viewPlane;
    Color                _bgColor;
    Handle<Camera>       _camera;
    Handle<Image>        _image;
    Handle<Octree>       _octree;
    Handle<Sampler>      _sampler;
    Handle<Tracer>       _tracer;
    Handle<AmbientLight> _ambientLight;
    std::vector<Handle<Plane>>    _planes;
    std::vector<Handle<Geometry>> _objects;
    std::vector<Handle<Light>>    _lights;

    // planes need to be separate because they will always be checked,
    // whereas all other objects will be spatially partitioned

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
    /// Gets the ambient light's color.
    /// </summary>
    const Color& GetAmbientColor() const;

    /// <summary>
    /// Gets the ambient light's radiance.
    /// </summary>
    Color GetAmbientRadiance() const;

    /// <summary>
    /// Gets the ambient light's radiance scale.
    /// </summary>
    real32 GetAmbientRadianceScale() const;

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
    /// Gets all of the lights in this scene.
    /// </summary>
    const std::vector<Handle<Light>>& GetLights() const;

    /// <summary>
    /// Gets the number of lights in this scene.
    /// </summary>
    uint32 GetLightCount() const;

    /// <summary>
    /// Gets all of the objects in this scene.
    /// </summary>
    const std::vector<Handle<Geometry>>& GetObjects() const;

    /// <summary>
    /// Gets the number of objects in this scene.
    /// </summary>
    uint32 GetObjectCount() const;

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
    /// <param name="sp">The shade point object to fill data with.</param>
    void HitObjects( const Ray& ray, ShadePoint& sp ) const;

    /// <summary>
    /// Shadow-hits all of the objects in this scene with the given ray.
    /// </summary>
    /// <param name="ray">The ray to hit with.</param>
    bool ShadowHitObjects( const Ray& ray ) const;

    /// <summary>
    /// Adds a plane to the scene.
    /// </summary>
    /// <param name="plane">The plane to add.</param>
    Handle<Plane> AddPlane( const Plane& plane );

    /// <summary>
    /// Adds a plane to the scene.
    /// </summary>
    /// <param name="point">A point the plane passes through.</param>
    /// <param name="normal">The plane's normal.</param>
    Handle<Plane> AddPlane( const Vector3& point, const Vector3& normal );

    /// <summary>
    /// Adds a plane to the scene.
    /// </summary>
    /// <param name="point">A point the plane passes through.</param>
    /// <param name="normal">The plane's normal.</param>
    /// <param name="material">The material of the plane.</param>
    template<class T> Handle<Plane> AddPlane( const Vector3& point, const Vector3& normal, const T& material );

    /// <summary>
    /// Adds a point light to the scene.
    /// </summary>
    /// <param name="position">The light's coordinates.</param>
    Handle<PointLight> AddPointLight( const Vector3& position );

    /// <summary>
    /// Adds a point light to the scene.
    /// </summary>
    /// <param name="x">The light's X coordinate.</param>
    /// <param name="y">The light's Y coordinate.</param>
    /// <param name="z">The light's Z coordinate.</param>
    Handle<PointLight> AddPointLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Adds a directional light to the scene.
    /// </summary>
    /// <param name="direction">The light's direction.</param>
    Handle<DirectionalLight> AddDirectionalLight( const Vector3& direction );

    /// <summary>
    /// Adds a directional light to the scene.
    /// </summary>
    /// <param name="x">The light's X direction.</param>
    /// <param name="y">The light's Y direction.</param>
    /// <param name="z">The light's Z direction.</param>
    Handle<DirectionalLight> AddDirectionalLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Adds a sphere to the scene.
    /// </summary>
    /// <param name="sphere">The sphere to add.</param>
    Handle<Sphere> AddSphere( const Sphere& sphere );

    /// <summary>
    /// Adds a sphere to the scene.
    /// </summary>
    /// <param name="center">The center of the sphere.</param>
    /// <param name="radius">The radius of the sphere.</param>
    Handle<Sphere> AddSphere( const Vector3& center, real64 radius );

    /// <summary>
    /// Adds a sphere to the scene.
    /// </summary>
    /// <param name="center">The center of the sphere.</param>
    /// <param name="radius">The radius of the sphere.</param>
    /// <param name="material">The material of the sphere.</param>
    template<class T> Handle<Sphere> AddSphere( const Vector3& center, real64 radius, const T& material );

    /// <summary>
    /// Builds the scene.
    /// </summary>
    /// <param name="hres">The horizontal resolution.</param>
    /// <param name="vres">The vertical resolution.</param>
    /// <param name="ps">The pixel size to use.</param>
    void Build( int32 hres, int32 vres, real32 ps );

    /// <summary>
    /// Builds the scene's octree for spatial partitioning.
    /// </summary>
    void BuildOctree();
    
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
    /// Sets this scene's ambient color.
    /// </summary>
    /// <param name=""></param>
    void SetAmbientColor( const Color& color );

    /// <summary>
    /// Sets this scene's ambient color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    void SetAmbientColor( real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets this scene's ambient radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    void SetAmbientRadianceScale( real32 ls );

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