#include <rex/Graphics/Camera.hxx>
#include <rex/Graphics/Color.hxx>
#include <rex/Graphics/Scene.hxx>

REX_NS_BEGIN

// create new camera
Camera::Camera()
{
    _up             = Vector3( 0.0, 1.0, 0.0 );
    _target         = Vector3( 0.0, 0.0, 1.0 );
    _viewPlaneDist  = real_t( 1000.0 );
}

// destroy camera
Camera::~Camera()
{
    _viewPlaneDist = real_t( 0.0 );
}

// get ortho X axis
const Vector3& Camera::GetOrthoX() const
{
    return _orthoU;
}

// get ortho Y axis
const Vector3& Camera::GetOrthoY() const
{
    return _orthoV;
}

// get ortho Z axis
const Vector3& Camera::GetOrthoZ() const
{
    return _orthoW;
}

// get camera position
const Vector3& Camera::GetPosition() const
{
    return _position;
}

// get ray to sample point
Vector3 Camera::GetRayDirection( const Vector2& sp ) const
{
    Vector3 dir = sp.X * _orthoU            // +x is right
                - sp.Y * _orthoV            // +y is up
                - _viewPlaneDist * _orthoW; // +z is out of screen (I think)
    return Vector3::Normalize( dir );
}

// get camera target
const Vector3& Camera::GetTarget() const
{
    return _target;
}

// calculate orthonormal basis vectors
void Camera::CalculateOrthonormalVectors()
{
    // calculate basis vectors
    _orthoW = Vector3::Normalize( _position - _target );
    _orthoU = Vector3::Normalize( Vector3::Cross( _up, _orthoW ) );
    _orthoV = Vector3::Cross( _orthoW, _orthoU );


    // handle the singularity if the camera if looking directly down
    if ( _position.X == _target.X &&
         _position.Z == _target.Z &&
         _position.Y >  _target.Y )
    {
        _orthoV = Vector3( 1.0, 0.0, 0.0 );
        _orthoW = Vector3( 0.0, 1.0, 0.0 );
        _orthoU = Vector3( 0.0, 0.0, 1.0 );
    }

    // handle the singularity if the camera is looking directly up
    if ( _position.X == _target.X &&
         _position.Z == _target.Z &&
         _position.Y <  _target.Y )
    {
        _orthoU = Vector3( 1.0,  0.0, 0.0 );
        _orthoW = Vector3( 0.0, -1.0, 0.0 );
        _orthoV = Vector3( 0.0,  0.0, 1.0 );
    }
}

// set camera position
void Camera::SetPosition( const Vector3& position )
{
    _position = position;
}

// set camera position
void Camera::SetPosition( real_t x, real_t y, real_t z )
{
    _position = Vector3( x, y, z );
}

// set camera target
void Camera::SetTarget( const Vector3& target )
{
    _target = target;
}

// set camera target
void Camera::SetTarget( real_t x, real_t y, real_t z )
{
    _target = Vector3( x, y, z );
}

// set camera up
void Camera::SetUp( const Vector3& up )
{
    _up = up;
}

// set camera up
void Camera::SetUp( real_t x, real_t y, real_t z )
{
    _up = Vector3( x, y, z );
}

// set camera view plane distance
void Camera::SetViewPlaneDistance( real_t dist )
{
    _viewPlaneDist = dist;
}

REX_NS_END