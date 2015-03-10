#ifndef __REX_GEOMETRY_INL
#define __REX_GEOMETRY_INL

#include "Geometry.hxx"

REX_NS_BEGIN

// create geometry w/ material
template<class T> Geometry::Geometry( const T& material )
{
    _material.reset( new T( material ) );
}

// get geometry bounds by population
inline void Geometry::GetBounds( BoundingBox& box ) const
{
    BoundingBox bounds = GetBounds();
    box.SetMin( bounds.GetMin() );
    box.SetMax( bounds.GetMax() );
}

// set geometry's material
template<class T> inline void Geometry::SetMaterial( const Handle<T>& material )
{
    _material.reset();
    _material = material;
}

// set geometry's material
template<class T> inline void Geometry::SetMaterial( const T& material )
{
    _material.reset( new T( material ) );
}

REX_NS_END

#endif