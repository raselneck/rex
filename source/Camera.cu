#include <rex/Graphics/Camera.hxx>
#include <rex/Graphics/Color.hxx>
#include <rex/Graphics/Scene.hxx>
#include <rex/Utility/Logger.hxx>

using namespace glm;

REX_NS_BEGIN

#define UnitXAxis vec3( 1, 0, 0 )
#define UnitYAxis vec3( 0, 1, 0 )
#define UnitZAxis vec3( 0, 0, 1 )

// create new camera
Camera::Camera()
    : _yaw          ( 0.0f )
    , _pitch        ( 0.0f )
    , _viewPlaneDist( 2000.0f )
    , _forward      ( UnitZAxis )
    , _up           ( UnitYAxis )
    , _right        ( UnitXAxis )
{
}

// destroy camera
Camera::~Camera()
{
    _viewPlaneDist = 0.0f;
}

// get camera position
const vec3& Camera::GetPosition() const
{
    return _position;
}

// get ray to sample point
vec3 Camera::GetRayDirection( const vec2& sp ) const
{
    vec3 dir = _right   * sp.x            // +x is right
             - _up      * sp.y            // +y is up
             - _forward * _viewPlaneDist; // +z is out of screen (I think)
    return normalize( dir );
}

// get local X axis
const vec3& Camera::GetLocalXAxis() const
{
    return _right;
}

// get local Y axis
const vec3& Camera::GetLocalYAxis() const
{
    return _up;
}

// get local Z axis
const vec3& Camera::GetLocalZAxis() const
{
    return _forward;
}

// look at the given target from the given position
void Camera::LookAt( const vec3& position, const vec3& target )
{
    // calculations taken from http://stackoverflow.com/a/4036151/1491629

    vec3 delta = normalize( position - target );
    SetPitch( atan2( delta.y, delta.z ) );
    SetYaw  ( atan2( delta.x, sqrt( delta.y * delta.y + delta.z * delta.z ) ) );

    _position = position;
}

// move the camera
void Camera::Move( const vec3& amount )
{
    _translation += amount;
}

// move the camera
void Camera::Move( real32 x, real32 y, real32 z )
{
    _translation.x += x;
    _translation.y += y;
    _translation.z += z;
}

// move to position
void Camera::MoveTo( const vec3& position )
{
    _position = position;
    _translation.x = _translation.y = _translation.z = 0.0f;
}

// move to position
void Camera::MoveTo( real32 x, real32 y, real32 z )
{
    _position.x = x;
    _position.y = y;
    _position.z = z;
    _translation.x = _translation.y = _translation.z = 0.0f;
}

// rotate the camera
void Camera::Rotate( const vec2& amount )
{
    SetPitch( _pitch + amount.x );
    SetYaw  ( _yaw   + amount.y );
}

// rotate the camera
void Camera::Rotate( real32 x, real32 y )
{
    SetPitch( _pitch + x );
    SetYaw  ( _yaw   + y );
}

// set pitch
void Camera::SetPitch( real32 pitch )
{
    const real32 halfPi = Math::HalfPi();

    _pitch = clamp( pitch, -halfPi, halfPi );
}

// set yaw
void Camera::SetYaw( real32 yaw )
{
    const real32 twoPi = Math::TwoPi();

    while ( yaw > twoPi ) yaw -= twoPi;
    while ( yaw < 0.0f  ) yaw += twoPi;

    _yaw = yaw;
}

// set view plane distance
void Camera::SetViewPlaneDistance( real32 dist )
{
    _viewPlaneDist = dist;
}

// update to account for changes
void Camera::Update()
{
    // calculate the rotation matrix
    _rotation = rotate( _pitch, UnitXAxis )
              * rotate( _yaw,   UnitYAxis );


    // calculate the actual transformation and new position
    _translation    = Math::Transform( _translation, _rotation );
    _position      += _translation;
    _translation.x  = _translation.y = _translation.z = 0.0f;


    // update the local coordinate axes
    // TODO : Account for when the camera is looking directly up and down?
    _right   = Math::Transform( UnitXAxis, _rotation );
    _up      = Math::Transform( UnitYAxis, _rotation );
    _forward = Math::Transform( UnitZAxis, _rotation );
}

REX_NS_END