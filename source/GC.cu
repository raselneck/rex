#include <rex/Utility/GC.hxx>

using namespace std;

REX_NS_BEGIN

GC GC::_instance;

// create the garbage collector instance
GC::GC()
{
}

// destroy the garbage collector instance
GC::~GC()
{
    // clear all of the device memory
    for ( auto& dm : _deviceMem )
    {
        cudaFree( dm );
    }
    _deviceMem.clear();

    // clear all of the host memory
    for ( auto& hm : _hostMem )
    {
        CleanupCallback cleanup = hm.second;
        void*           data = hm.first;

        cleanup( data );
    }
    _hostMem.clear();
}

// register device memory
void GC::RegisterDeviceMemory( void* mem )
{
    _instance._deviceMem.push_back( mem );
}

// unregister device memory
void GC::UnregisterDeviceMemory( void* mem )
{
    for ( uint_t i = 0; i < _instance._deviceMem.size(); ++i )
    {
        if ( _instance._deviceMem[ i ] == mem )
        {
            _instance._deviceMem.erase( _instance._deviceMem.begin() + i );
            break;
        }
    }
}

REX_NS_END