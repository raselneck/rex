#include <rex/Graphics/Camera.hxx>
#include <rex/Graphics/Color.hxx>
#include <rex/Graphics/Scene.hxx>

REX_NS_BEGIN

// create new camera
Camera::Camera()
{
    _up             = vec3( 0.0, 1.0, 0.0 );
    _target         = vec3( 0.0, 0.0, 1.0 );
    _viewPlaneDist  = real32( 1000.0 );
}

// destroy camera
Camera::~Camera()
{
    _viewPlaneDist = real32( 0.0 );
}

// get ortho X axis
const vec3& Camera::GetOrthoX() const
{
    return _orthoU;
}

// get ortho Y axis
const vec3& Camera::GetOrthoY() const
{
    return _orthoV;
}

// get ortho Z axis
const vec3& Camera::GetOrthoZ() const
{
    return _orthoW;
}

// get camera position
const vec3& Camera::GetPosition() const
{
    return _position;
}

// get ray to sample point
vec3 Camera::GetRayDirection( const vec2& sp ) const
{
    vec3 dir = sp.x * _orthoU            // +x is right
             - sp.y * _orthoV            // +y is up
             + _viewPlaneDist * _orthoW; // +z is out of screen (I think)
    return glm::normalize( dir );
}

// get camera target
const vec3& Camera::GetTarget() const
{
    return _target;
}

// calculate orthonormal basis vectors
void Camera::CalculateOrthonormalVectors()
{
    // calculate basis vectors
    _orthoW = glm::normalize( _position - _target );
    _orthoU = glm::normalize( glm::cross( _up, _orthoW ) );
    _orthoV = glm::cross( _orthoW, _orthoU );


    // handle the singularity if the camera if looking directly down
    if ( _position.x == _target.x &&
         _position.z == _target.z &&
         _position.y >  _target.y )
    {
        _orthoV = vec3( 1.0f, 0.0f, 0.0f );
        _orthoW = vec3( 0.0f, 1.0f, 0.0f );
        _orthoU = vec3( 0.0f, 0.0f, 1.0f );
    }

    // handle the singularity if the camera is looking directly up
    if ( _position.x == _target.x &&
         _position.z == _target.z &&
         _position.y <  _target.y )
    {
        _orthoU = vec3( 1.0f,  0.0f, 0.0f );
        _orthoW = vec3( 0.0f, -1.0f, 0.0f );
        _orthoV = vec3( 0.0f,  0.0f, 1.0f );
    }
}

// set camera position
void Camera::SetPosition( const vec3& position )
{
    _position = position;
}

// set camera position
void Camera::SetPosition( real32 x, real32 y, real32 z )
{
    _position = vec3( x, y, z );
}

// set camera target
void Camera::SetTarget( const vec3& target )
{
    _target = target;
}

// set camera target
void Camera::SetTarget( real32 x, real32 y, real32 z )
{
    _target = vec3( x, y, z );
}

// set camera up
void Camera::SetUp( const vec3& up )
{
    _up = up;
}

// set camera up
void Camera::SetUp( real32 x, real32 y, real32 z )
{
    _up = vec3( x, y, z );
}

// set camera view plane distance
void Camera::SetViewPlaneDistance( real32 dist )
{
    _viewPlaneDist = dist;
}

REX_NS_END