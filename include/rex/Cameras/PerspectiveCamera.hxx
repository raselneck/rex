#ifndef __REX_PERSPECTIVECAMERA_HXX
#define __REX_PERSPECTIVECAMERA_HXX

#include "../Config.hxx"
#include "../Cameras/Camera.hxx"
#include "../Utility/Vector2.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a perspective camera.
/// </summary>
class PerspectiveCamera : public Camera
{
    real32 _viewPlaneDistance;
    real32 _zoomAmount;

    /// <summary>
    /// Gets the ray direction to the given sample point.
    /// </summary>
    /// <param name=""></param>
    Vector3 GetRayDirection( const Vector2& sp ) const;

public:
    /// <summary>
    /// Creates a new perspective camera.
    /// </summary>
    PerspectiveCamera();

    /// <summary>
    /// Destroys this perspective camera.
    /// </summary>
    virtual ~PerspectiveCamera();

    /// <summary>
    /// Renders the given scene.
    /// </summary>
    /// <param name="scene">The scene to render.</param>
    virtual void Render( Scene& scene );

    /// <summary>
    /// Sets this perspective camera's view plane distance.
    /// </summary>
    /// <param name="dist">The new view plane distance.</param>
    void SetViewPlaneDistance( real32 dist );

    /// <summary>
    /// Sets this perspective camera's zoom amount.
    /// </summary>
    /// <param name="zoom">The new zoom amount.</param>
    void SetZoomAmount( real32 zoom );
};

REX_NS_END

#endif