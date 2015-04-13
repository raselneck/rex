#pragma once

#include "../Config.hxx"
#include "../Math/Vector2.hxx"
#include "../Math/Vector3.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a camera.
/// </summary>
class Camera
{
    Vector3 _position;
    Vector3 _target;
    Vector3 _up;
    Vector3 _orthoU;
    Vector3 _orthoV;
    Vector3 _orthoW;
    real_t  _viewPlaneDist;

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
    __both__ const Vector3& GetPosition() const;

    /// <summary>
    /// Gets the ray direction to the given sample point.
    /// </summary>
    /// <param name="sp">The sample point.</param>
    __both__ Vector3 GetRayDirection( const Vector2& sp ) const;

    /// <summary>
    /// Gets this camera's target.
    /// </summary>
    __both__ const Vector3& GetTarget() const;

    /// <summary>
    /// Calculates the orthonormal basis vectors.
    /// </summary>
    __both__ void CalculateOrthonormalVectors();

    /// <summary>
    /// Sets this camera's position.
    /// </summary>
    /// <param name="position">The new position.</param>
    __both__ void SetPosition( const Vector3& position );

    /// <summary>
    /// Sets this camera's position.
    /// </summary>
    /// <param name="x">The new position's X component.</param>
    /// <param name="y">The new position's Y component.</param>
    /// <param name="z">The new position's Z component.</param>
    __both__ void SetPosition( real_t x, real_t y, real_t z );

    /// <summary>
    /// Sets this camera's target.
    /// </summary>
    /// <param name="target">The new target.</param>
    __both__ void SetTarget( const Vector3& target );

    /// <summary>
    /// Sets this camera's target.
    /// </summary>
    /// <param name="x">The new target's X component.</param>
    /// <param name="y">The new target's Y component.</param>
    /// <param name="z">The new target's Z component.</param>
    __both__ void SetTarget( real_t x, real_t y, real_t z );

    /// <summary>
    /// Sets the relative up vector.
    /// </summary>
    /// <param name="up">The new up vector.</param>
    __both__ void SetUp( const Vector3& up );

    /// <summary>
    /// Sets the relative up vector.
    /// </summary>
    /// <param name="x">The new up vector's X component.</param>
    /// <param name="y">The new up vector's Y component.</param>
    /// <param name="z">The new up vector's Z component.</param>
    __both__ void SetUp( real_t x, real_t y, real_t z );

    /// <summary>
    /// Sets the view plane's distance.
    /// </summary>
    /// <param name="dist">The new distance.</param>
    __both__ void SetViewPlaneDistance( real_t dist );
};

REX_NS_END