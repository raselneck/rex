#pragma once

#include "../Config.hxx"
#include "../CUDA/DeviceList.hxx"
#include "../GL/GLTexture2D.hxx"
#include "../GL/GLWindow.hxx"
#include "../Utility/Image.hxx"
#include "Geometry/Octree.hxx"
#include "Lights/AmbientLight.hxx"
#include "Camera.hxx"
#include "ShadePoint.hxx"
#include "ViewPlane.hxx"

struct GLFWwindow; // forward declare

REX_NS_BEGIN

/// <summary>
/// An enumeration of possible render modes for the scene.
/// </summary>
enum class SceneRenderMode
{
    ToImage,
    ToOpenGL
};

/// <summary>
/// Defines a scene.
/// </summary>
class Scene
{
    REX_NONCOPYABLE_CLASS( Scene )

    const SceneRenderMode  _renderMode;
    ViewPlane              _viewPlane;
    Color                  _backgroundColor;
    Camera                 _camera;
    DeviceList<Light*>*    _lights;
    AmbientLight*          _ambientLight;
    DeviceList<Geometry*>* _geometry;
    Octree*                _octree;
    GLWindow*              _window;
    GLTexture2D*           _texture;
    Image*                 _image;

    /// <summary>
    /// Performs pre-render actions.
    /// </summary>
    __host__ bool OnPreRender();

    /// <summary>
    /// Performs post-render actions.
    /// </summary>
    __host__ bool OnPostRender();

    /// <summary>
    /// Updates the camera based on user input.
    /// <summary>
    /// <param name="dt">The time since the last frame.</param>
    /// <remarks>
    /// Only called when rendering to an OpenGL window.
    /// </remarks>
    __host__ void UpdateCamera( real64 dt );

    /// <summary>
    /// Disposes of this scene.
    /// </summary>
    __host__ void Dispose();

    /// <summary>
    /// The GLFW window key press callback.
    /// <summary>
    /// <param name="dt">The time since the last frame.</param>
    __host__ static void OnKeyPress( GLFWwindow* window, int key, int scancode, int action, int mods );

public:
    /// <summary>
    /// Creates a new scene.
    /// </summary>
    /// <param name="renderMode">The render mode to use.</param>
    __host__ Scene( SceneRenderMode renderMode );

    /// <summary>
    /// Destroys this scene.
    /// </summary>
    __host__ ~Scene();

    /// <summary>
    /// Saves this scene's image.
    /// </summary>
    /// <param name="fname">The file name.</param>
    __host__ void SaveImage( const char* fname ) const;

    /// <summary>
    /// Builds this scene.
    /// </summary>
    /// <param name="width">The width of the rendered scene. The maximum width is 2048.</param>
    /// <param name="height">The height of the rendered scene. The maximum height is 2048.</param>
    __host__ bool Build( uint16 width, uint16 height );

    /// <summary>
    /// Builds this scene.
    /// </summary>
    /// <param name="width">The width of the rendered scene.</param>
    /// <param name="height">The height of the rendered scene.</param>
    /// <param name="samples">The sample count to render with.</param>
    __host__ bool Build( uint16 width, uint16 height, int32 samples );

    /// <summary>
    /// Builds this scene.
    /// </summary>
    /// <param name="width">The width of the rendered scene.</param>
    /// <param name="height">The height of the rendered scene.</param>
    /// <param name="samples">The sample count to render with.</param>
    /// <param name="fullscreen">Whether or not to be building for a fullscreen window (only applies when rendering to OpenGL).</param>
    __host__ bool Build( uint16 width, uint16 height, int32 samples, bool fullscreen );

    /// <summary>
    /// Gets this scene's camera.
    /// </summary>
    __host__ Camera& GetCamera();

    /// <summary>
    /// Renders this scene.
    /// </summary>
    __host__ void Render();
};

REX_NS_END