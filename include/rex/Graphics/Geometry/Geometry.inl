#pragma once

#include "../../Utility/Logger.hxx"

REX_NS_BEGIN

// create a new piece of geometry
template<typename T> Geometry::Geometry( const T& material )
    : _hMaterial( nullptr ),
      _dMaterial( nullptr )
{
    SetMaterial<T>( material );
}

// set the material of this piece of geometry
template<typename T> void Geometry::SetMaterial( const T& material )
{
    // ensure the material is valid (compiler will catch if it's not)
    MaterialType type = material.GetType();
    (void)type;

    // delete the old material if we have one
    if ( _hMaterial )
    {
        delete _hMaterial;
    }

    // set the new material
    _hMaterial = reinterpret_cast<Material*>( new T() );
    _dMaterial = _hMaterial->GetOnDevice();
}

REX_NS_END