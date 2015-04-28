#pragma once

#include "../Config.hxx"
#include "../Math/Math.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a camera.
/// </summary>
class Camera
{
    mat4 _rotation;
    vec3 _position;
    vec3 _translation;
    vec3 _forward;
    vec3 _up;
    vec3 _right;
    real32 _yaw;
    real32 _pitch;
    real32 _viewPlaneDist;

public:
    /// <summary>
    /// Creates a new camera.
    /// </summary>
    __both__ Camera();

    /// <summary>
    /// Destroys this camera.
    /// </summary>
    __both__ ~Camera();

    /// <summary>
    /// Gets this camera's position.
    /// </summary>
    __both__ const vec3& GetPosition() const;

    /// <summary>
    /// Gets the ray direction to the given sample point.
    /// </summary>
    /// <param name="sp">The sample point.</param>
    __both__ vec3 GetRayDirection( const vec2& sp ) const;

    /// <summary>
    /// Gets this camera's local X axis.
    /// </summary>
    __both__ const vec3& GetLocalXAxis() const;

    /// <summary>
    /// Gets this camera's local Y axis.
    /// </summary>
    __both__ const vec3& GetLocalYAxis() const;

    /// <summary>
    /// Gets this camera's local Z axis.
    /// </summary>
    __both__ const vec3& GetLocalZAxis() const;

    /// <summary>
    /// Moves this camera to the given position and rotates it to look at the given target.
    /// </summary>
    /// <param name="position">The position to move to.</param>
    /// <param name="target">The target to look at.</param>
    __both__ void LookAt( const vec3& position, const vec3& target );

    /// <summary>
    /// Moves this camera.
    /// </summary>
    /// <param name="amount">The amount to move.</param>
    __both__ void Move( const vec3& amount );

    /// <summary>
    /// Moves this camera.
    /// </summary>
    /// <param name="x">The X amount to move.</param>
    /// <param name="y">The Y amount to move.</param>
    /// <param name="z">The Z amount to move.</param>
    __both__ void Move( real32 x, real32 y, real32 z );

    /// <summary>
    /// Moves this camera to the given position.
    /// </summary>
    /// <param name="position">The new position.</param>
    __both__ void MoveTo( const vec3& position );

    /// <summary>
    /// Moves this camera to the given position.
    /// </summary>
    /// <param name="x">The new position's X coordinate.</param>
    /// <param name="y">The new position's Y coordinate.</param>
    /// <param name="z">The new position's Z coordinate.</param>
    __both__ void MoveTo( real32 x, real32 y, real32 z );

    /// <summary>
    /// Rotates this camera.
    /// </summary>
    /// <param name="amount">The amount to rotate.</param>
    __both__ void Rotate( const vec2& amount );

    /// <summary>
    /// Rotates this camera.
    /// </summary>
    /// <param name="x">The X amount to rotate.</param>
    /// <param name="y">The Y amount to rotate.</param>
    __both__ void Rotate( real32 x, real32 y );

    /// <summary>
    /// Sets this camera's pitch value (i.e. the rotation about the X-axis).
    /// </summary>
    /// <param name="pitch">The new pitch value, in radians.</param>
    __both__ void SetPitch( real32 pitch );

    /// <summary>
    /// Sets this camera's yaw value (i.e. the rotation about the Y-axis).
    /// </summary>
    /// <param name="yaw">The new yaw value, in radians.</param>
    __both__ void SetYaw( real32 yaw );

    /// <summary>
    /// Sets the view plane's distance.
    /// </summary>
    /// <param name="dist">The new distance.</param>
    __both__ void SetViewPlaneDistance( real32 dist );

    /// <summary>
    /// Updates this camera to account for any changes.
    /// </summary>
    __both__ void Update();
};

REX_NS_END