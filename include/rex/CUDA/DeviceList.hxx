#pragma once

#include "../Config.hxx"
#include "../CUDA.hxx"
#include <stdlib.h>

REX_NS_BEGIN

/// <summary>
/// Defines a rudimentary, resizable device list.
/// </summary>
template<typename T> class DeviceList
{
    T*     _items;
    uint32 _size;

public:
    /// <summary>
    /// Creates a new device list.
    /// </summary>
    __both__ DeviceList();

    /// <summary>
    /// Destroys this device list.
    /// </summary>
    __both__ ~DeviceList();

    /// <summary>
    /// Gets the number of items in this list.
    /// </summary>
    __device__ uint32 GetSize() const;

    /// <summary>
    /// Adds the given item to this list.
    /// </summary>
    /// <param name="item">The item to add.</param>
    __device__ void Add( const T& item );

    /// <summary>
    /// Removes the item at the given index.
    /// </summary>
    /// <param name="index">The index of the item to remove.</param>
    __device__ void Remove( uint32 index );

    /// <summary>
    /// Resizes this list.
    /// </summary>
    /// <param name="size">The new size.</param>
    __device__ void Resize( uint32 size );

    /// <summary>
    /// Gets the item at the given index.
    /// </summary>
    /// <param name="index">The index.</param>
    __device__ const T& operator[]( uint32 index ) const;

    /// <summary>
    /// Gets the item at the given index.
    /// </summary>
    /// <param name="index">The index.</param>
    __device__ T& operator[]( uint32 index );
};

REX_NS_END

#include "DeviceList.inl"