#pragma once

#include "../../Utility/GC.hxx"

REX_NS_BEGIN

// create new sphere
template<typename T> Sphere::Sphere( const T& material )
    : Geometry( material ),
      _radius( 0.0 ),
      _invRadius( 0.0 )
{
    _dThis = GC::DeviceAlloc<Sphere>( *this );
}

// create new sphere
template<typename T> Sphere::Sphere( const T& material, const Vector3& center, real64 radius )
    : Geometry( material ),
      _center( center ),
      _radius( radius ),
      _invRadius( ( radius == 0.0f) ? 0.0f : 1.0 / radius )
{
    _dThis = GC::DeviceAlloc<Sphere>( *this );
}

REX_NS_END