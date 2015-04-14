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
    uint_t _size;

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
    __device__ uint_t GetSize() const;

    /// <summary>
    /// Adds the given item to this list.
    /// </summary>
    /// <param name="item">The item to add.</param>
    __device__ void Add( const T& item );

    /// <summary>
    /// Removes the item at the given index.
    /// </summary>
    /// <param name="index">The index of the item to remove.</param>
    __device__ void Remove( uint_t index );

    /// <summary>
    /// Resizes this list.
    /// </summary>
    /// <param name="size">The new size.</param>
    __device__ void Resize( uint_t size );

    /// <summary>
    /// Gets the item at the given index.
    /// </summary>
    /// <param name="index">The index.</param>
    __device__ const T& operator[]( uint_t index ) const;

    /// <summary>
    /// Gets the item at the given index.
    /// </summary>
    /// <param name="index">The index.</param>
    __device__ T& operator[]( uint_t index );
};

REX_NS_END

#include "DeviceList.inl"