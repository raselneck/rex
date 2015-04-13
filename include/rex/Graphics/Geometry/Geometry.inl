#pragma once

REX_NS_BEGIN

// create a new piece of geometry
template<typename T> __device__ Geometry::Geometry( GeometryType type, const T& material )
    : _material    ( nullptr ),
      _geometryType( type )
{
    SetMaterial<T>( material );
}

// set the material of this piece of geometry
template<typename T> __device__ void Geometry::SetMaterial( const T& material )
{
    // ensure the material is valid (compiler will catch if it's not)
    MaterialType type = material.GetType();
    (void)type;

    // delete the old material if we have one
    if ( _material )
    {
        delete _material;
        _material = nullptr;
    }

    // set the new material
    _material = material.Copy();
}

REX_NS_END