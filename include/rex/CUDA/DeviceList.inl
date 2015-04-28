#include "../Math/Math.hxx"

REX_NS_BEGIN

// create new device list
template<typename T> DeviceList<T>::DeviceList()
    : _items( nullptr ),
      _size ( 0 )
{
}

// destroy device list
template<typename T> DeviceList<T>::~DeviceList()
{
    if ( _items )
    {
        delete[] _items;
        _items = nullptr;
    }

    _size = 0;
}

// get item in list
template<typename T> __device__ const T& DeviceList<T>::Get( uint32 index ) const
{
    return _items[ index ];
}

// get size of device list
template<typename T> __device__ uint32 DeviceList<T>::GetSize() const
{
    return _size;
}

// add item to list
template<typename T> __device__ void DeviceList<T>::Add( const T& item )
{
    Resize( _size + 1 );
    this->operator[]( _size - 1 ) = item;
}

// get item in list
template<typename T> __device__ T& DeviceList<T>::Get( uint32 index )
{
    return _items[ index ];
}

// remove item from the list
template<typename T> __device__ void DeviceList<T>::Remove( uint32 index )
{
    for ( uint32 i = index; i < _size - 1; ++i )
    {
        _items[ i ] = _items[ i + 1 ];
    }
    if ( index < _size )
    {
        Resize( _size - 1 );
    }
}

// resize list
template<typename T> __device__ void DeviceList<T>::Resize( uint32 size )
{
    // create the new items
    T*     newItems = new T[ size ];
    uint32 toSize   = Math::Min( _size, size );

    // copy over the data
    for ( uint32 i = 0; i < toSize; ++i )
    {
        newItems[ i ] = _items[ i ];
    }

    // delete the old items if necessary
    if ( _items )
    {
        delete[] _items;
        _items = nullptr;
    }

    // update our data members
    _items = newItems;
    _size  = size;
}

// get item in list
template<typename T> __device__ const T& DeviceList<T>::operator[]( uint32 index ) const
{
    return Get( index );
}

// get item in list
template<typename T> __device__ T& DeviceList<T>::operator[]( uint32 index )
{
    return Get( index );
}

REX_NS_END