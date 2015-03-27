#include <rex/Graphics/Lights/LightCollection.hxx>
#include <rex/Utility/GC.hxx>

REX_NS_BEGIN

// create new light collection
LightCollection::LightCollection()
{
}

// destroy this light collection
LightCollection::LightCollection()
{
}

// get light count
uint32 LightCollection::GetLightCount() const
{
    return static_cast<uint32>( _hLights.size() );
}

REX_NS_END