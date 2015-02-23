#ifndef __REX_GEOMETRY_INL
#define __REX_GEOMETRY_INL

#include "Geometry.hxx"

REX_NS_BEGIN

// create geometry w/ material
template<class T> Geometry::Geometry( const T& material )
{
    _material.reset( new T( material ) );
}

// set geometry's material
template<class T> void Geometry::SetMaterial( const T& material )
{
    _material.reset( new T( material ) );
}

REX_NS_END

#endif