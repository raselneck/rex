#pragma once

#include "../Config.hxx"
#include "../Math/Math.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a camera.
/// </summary>
class Camera
{
    vec3 _position;
    vec3 _target;
    vec3 _up;
    vec3 _orthoU;
    vec3 _orthoV;
    vec3 _orthoW;
    real32  _viewPlaneDist;

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
    /// Gets this camera's target.
    /// </summary>
    __both__ const vec3& GetTarget() const;

    /// <summary>
    /// Gets this camera's orthogonal X axis.
    /// </summary>
    __both__ const vec3& GetOrthoX() const;

    /// <summary>
    /// Gets this camera's orthogonal Y axis.
    /// </summary>
    __both__ const vec3& GetOrthoY() const;

    /// <summary>
    /// Gets this camera's orthogonal Z axis.
    /// </summary>
    __both__ const vec3& GetOrthoZ() const;

    /// <summary>
    /// Calculates the orthonormal basis vectors.
    /// </summary>
    __both__ void CalculateOrthonormalVectors();

    /// <summary>
    /// Sets this camera's position.
    /// </summary>
    /// <param name="position">The new position.</param>
    __both__ void SetPosition( const vec3& position );

    /// <summary>
    /// Sets this camera's position.
    /// </summary>
    /// <param name="x">The new position's X component.</param>
    /// <param name="y">The new position's Y component.</param>
    /// <param name="z">The new position's Z component.</param>
    __both__ void SetPosition( real32 x, real32 y, real32 z );

    /// <summary>
    /// Sets this camera's target.
    /// </summary>
    /// <param name="target">The new target.</param>
    __both__ void SetTarget( const vec3& target );

    /// <summary>
    /// Sets this camera's target.
    /// </summary>
    /// <param name="x">The new target's X component.</param>
    /// <param name="y">The new target's Y component.</param>
    /// <param name="z">The new target's Z component.</param>
    __both__ void SetTarget( real32 x, real32 y, real32 z );

    /// <summary>
    /// Sets the relative up vector.
    /// </summary>
    /// <param name="up">The new up vector.</param>
    __both__ void SetUp( const vec3& up );

    /// <summary>
    /// Sets the relative up vector.
    /// </summary>
    /// <param name="x">The new up vector's X component.</param>
    /// <param name="y">The new up vector's Y component.</param>
    /// <param name="z">The new up vector's Z component.</param>
    __both__ void SetUp( real32 x, real32 y, real32 z );

    /// <summary>
    /// Sets the view plane's distance.
    /// </summary>
    /// <param name="dist">The new distance.</param>
    __both__ void SetViewPlaneDistance( real32 dist );
};

REX_NS_END