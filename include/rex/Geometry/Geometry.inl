#ifndef __REX_GEOMETRY_INL
#define __REX_GEOMETRY_INL

#include "Geometry.hxx"

REX_NS_BEGIN

// create geometry w/ material
template<class T> Geometry::Geometry( const T& material )
{
    _material.reset( new T( material ) );
}

// create geometry w/ material handle
template<class T> Geometry::Geometry( const Handle<T>& material )
{
    _material = material;
}

// set geometry's material
inline void Geometry::SetMaterial( const Handle<Material>& material )
{
    _material.reset();
    _material = material;
}

// set geometry's material
inline void Geometry::SetMaterial( const Material& material )
{
    _material = material.Copy();
}

REX_NS_END

#endif