#include <rex/Cameras/Camera.hxx>

REX_NS_BEGIN

// new camera
Camera::Camera()
{
    SetUp( 0.0, 1.0, 0.0 );
}

// destroy camera
Camera::~Camera()
{
}

// calculate orthonormal basis vectors
void Camera::CalculateUVW()
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

// set position
void Camera::SetPosition( const Vector3& position )
{
    _position = position;
}

// set position
void Camera::SetPosition( real64 x, real64 y, real64 z )
{
    _position.X = x;
    _position.Y = y;
    _position.Z = z;
}

// set target
void Camera::SetTarget( const Vector3& target )
{
    _target = target;
}

// set target
void Camera::SetTarget( real64 x, real64 y, real64 z )
{
    _target.X = x;
    _target.Y = y;
    _target.Z = z;
}

// set up vector
void Camera::SetUp( const Vector3& up )
{
    _up = up;
}

// set up vector
void Camera::SetUp( real64 x, real64 y, real64 z )
{
    _up.X = x;
    _up.Y = y;
    _up.Z = z;
}

REX_NS_END