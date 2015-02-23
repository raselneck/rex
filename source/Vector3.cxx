#include <rex/Utility/Vector3.hxx>

REX_NS_BEGIN

// new 3D vector
Vector3::Vector3()
    : Vector3( 0.0, 0.0, 0.0 )
{
}

// new 3D vector
Vector3::Vector3( real64 all )
    : Vector3( all, all, all )
{
}

// new 3D vector
Vector3::Vector3( real64 x, real64 y, real64 z )
    : X( x ),
      Y( y ),
      Z( z )
{
}

// destroy 3D vector
Vector3::~Vector3()
{
    X = Y = Z = 0.0;
}

REX_NS_END