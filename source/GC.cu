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
    ReleaseDeviceMemory();

    // clear all of the host memory
    for ( auto& hm : _hostMem )
    {
        CleanupCallback cleanup = hm.second;
        void*           data = hm.first;

        cleanup( data );
    }
    _hostMem.clear();
}

// free all device memory
void GC::ReleaseDeviceMemory()
{
    // clear all of the device memory
    for ( auto& dm : _instance._deviceMem )
    {
        cudaFree( dm );
    }
    _instance._deviceMem.clear();
}

REX_NS_END