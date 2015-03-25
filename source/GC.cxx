#include <rex/Utility/GC.hxx>

using namespace std;

REX_NS_BEGIN

vector<pair<void*, GC::CleanupCallback>> GC::_hostMemory;
vector<void*> GC::_deviceMemory;
GC GC::_instance;

// create the garbage collector instance
GC::GC()
{
}

// destroy the garbage collector instance
GC::~GC()
{
    DisposeDevice();
    DisposeHost();
}

// dispose of host memory
void GC::DisposeHost()
{
    for ( auto& hm : _hostMemory )
    {
        hm.second( hm.first );
    }
    _hostMemory.clear();
}

// dispose of device memory
void GC::DisposeDevice()
{
    for ( auto& dm : _deviceMemory )
    {
        cudaCheck( cudaFree( dm ) );
    }
    _deviceMemory.clear();
}

REX_NS_END