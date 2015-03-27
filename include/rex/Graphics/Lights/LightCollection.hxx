#pragma once

#include "Light.hxx"
#include "AmbientLight.hxx"
#include "DirectionalLight.hxx"
#include "PointLight.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines a light collection.
/// </summary>
class LightCollection
{
    REX_NONCOPYABLE_CLASS( LightCollection );

    std::vector<Light*> _hLights;
    AmbientLight*       _hAmbientLight;
    Light**             _dLights;

public:
    /// <summary>
    /// Creates a new light collection.
    /// </summary>
    __host__ LightCollection();

    /// <summary>
    /// Destroys this light collection.
    /// </summary>
    __host__ ~LightCollection();

    /// <summary>
    /// Gets the number of lights in this collection.
    /// </summary>
    __host__ uint32 GetLightCount() const;

    /// <summary>
    /// Gets the array of lights in this collection on the device.
    /// </summary>
    __host__ Light** GetLightDeviceArray() const;
};

REX_NS_END