#include "Logger.hxx"

REX_NS_BEGIN

// the default host cleanup callback
template<typename T> void GC::HostCleanupCallback( void* data )
{
    delete static_cast<T*>( data );
}

// allocate memory for type on host
template<typename T, typename ... Args> T* GC::HostAlloc( const Args& ... args )
{
    // attempt to allocate the memory
    T* object = new ( std::nothrow ) T( args... );

    // if the memory was allocated, register it
    if ( object )
    {
        CleanupCallback cleanup = HostCleanupCallback<T>;
        void*           memory = static_cast<void*>( object );

        _instance._hostMem.push_back( std::make_pair( memory, cleanup ) );
    }
    else
    {
        REX_DEBUG_LOG( "Failed to allocate host memory for type ", REX_XSTRINGIFY( T ), "." );
    }

    // return the memory
    return object;
}

// allocate memory for an object on the device
template<typename T> T* GC::DeviceAlloc( const T& source )
{
    // attempt to allocate the memory
    T*          memory = nullptr;
    cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &memory ), sizeof( T ) );

    // if the memory was allocated, register it and set it
    if ( err == cudaSuccess )
    {
        // register memory
        _instance._deviceMem.push_back( memory );

        // copy it
        err = cudaMemcpy( memory, &source, sizeof( T ), cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            REX_DEBUG_LOG( "Allocated device memory for type ", REX_XSTRINGIFY( T ), " but failed to copy from source." );
        }
    }
    else
    {
        REX_DEBUG_LOG( "Failed to allocate device memory for type ", REX_XSTRINGIFY( T ), "." );
    }

    // return the memory
    return memory;
}

// allocate space for an array
template<typename T> T* GC::DeviceAllocArray( uint32 count )
{
    // attempt to allocate the memory
    T*          memory = nullptr;
    cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &memory ), sizeof( T ) * count );

    // if the memory was allocated, register it
    if ( err == cudaSuccess )
    {
        // register memory
        _instance._deviceMem.push_back( memory );
    }
    else
    {
        REX_DEBUG_LOG( "Failed to allocate device memory for ", REX_XSTRINGIFY( T ), " array." );
    }

    // return the memory
    return memory;
}

// allocate space for an array and copy data from a source
template<typename T> T* GC::DeviceAllocArray( uint32 count, const T* source )
{
    // allocate the memory
    T* memory = DeviceAllocArray<T>( count );

    // if the memory was allocated, copy over the data
    if ( memory )
    {
        cudaError_t err = cudaMemcpy( memory, source, count * sizeof( T ), cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            REX_DEBUG_LOG( "Allocated array of ", REX_XSTRINGIFY( T ), ", but failed to copy from source." );
        }
    }

    // return the memory
    return memory;
}

REX_NS_END