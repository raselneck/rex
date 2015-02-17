#ifndef __REX_CAMERA_HXX
#define __REX_CAMERA_HXX

#include "Config.hxx"
#include "Vector3.hxx"

REX_NS_BEGIN

class Scene;

/// <summary>
/// Defines the base for all cameras.
/// </summary>
class Camera
{
protected:
    Vector3 _position;
    Vector3 _target;
    Vector3 _up;
    Vector3 _orthoU; // orthonormal basis vector
    Vector3 _orthoV; // orthonormal basis vector
    Vector3 _orthoW; // orthonormal basis vector

    /// <summary>
    /// Calculates the new orthonormal basis vectors.
    /// </summary>
    void CalculateUVW();

public:
    /// <summary>
    /// Creates a new camera.
    /// </summary>
    Camera();

    /// <summary>
    /// Destroys this camera.
    /// </summary>
    virtual ~Camera();

    /// <summary>
    /// Renders the given scene.
    /// </summary>
    /// <param name="scene">The scene to render.</param>
    virtual void Render( Scene& scene ) = 0;

    /// <summary>
    /// Sets this camera's position.
    /// </summary>
    /// <param name="position">The new position.</param>
    void SetPosition( const Vector3& position );

    /// <summary>
    /// Sets this camera's position.
    /// </summary>
    /// <param name="x">The new X coordinate.</param>
    /// <param name="y">The new Y coordinate.</param>
    /// <param name="z">The new Z coordinate.</param>
    void SetPosition( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets this camera's target.
    /// </summary>
    /// <param name="target">The new target.</param>
    void SetTarget( const Vector3& target );

    /// <summary>
    /// Sets this camera's target.
    /// </summary>
    /// <param name="x">The new target's X coordinate.</param>
    /// <param name="y">The new target's Y coordinate.</param>
    /// <param name="z">The new target's Z coordinate.</param>
    void SetTarget( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets this camera's up vector.
    /// </summary>
    /// <param name="up">The new up vector.</param>
    void SetUp( const Vector3& up );

    /// <summary>
    /// Sets this camera's up vector.
    /// </summary>
    /// <param name="x">The new up vector's X coordinate.</param>
    /// <param name="y">The new up vector's Y coordinate.</param>
    /// <param name="z">The new up vector's Z coordinate.</param>
    void SetUp( real64 x, real64 y, real64 z );
};

REX_NS_END

#endif