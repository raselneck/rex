#ifndef __REX_MESH_INL
#define __REX_MESH_INL

#include "Mesh.hxx"

REX_NS_BEGIN

// set mesh material
template<class T> void Mesh::SetMaterial( const Handle<T>& material )
{
    _material = material;

    for ( auto iter = _triangles.begin(); iter != _triangles.end(); ++iter )
    {
        Handle<Triangle>& tri = *iter;
        tri->SetMaterial( _material );
    }
}

// set mesh material
template<class T> void Mesh::SetMaterial( const T& material )
{
    Handle<T> mat( new T( material ) );
    SetMaterial( mat );
}

REX_NS_END

#endif