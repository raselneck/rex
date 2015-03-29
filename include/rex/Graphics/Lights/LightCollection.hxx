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

    /// <summary>
    /// Updates the device array.
    /// </summary>
    __host__ void UpdateDeviceArray();

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
    /// Gets the ambient light.
    /// </summary>
    __host__ const AmbientLight* GetAmbientLight() const;

    /// <summary>
    /// Gets the number of lights in this collection.
    /// </summary>
    __host__ uint32 GetLightCount() const;

    /// <summary>
    /// Gets the array of lights in this collection.
    /// </summary>
    __host__ const std::vector<Light*>& GetLights() const;

    /// <summary>
    /// Gets the array of lights in this collection on the device.
    /// </summary>
    __host__ const Light** GetDeviceLights() const;

    /// <summary>
    /// Adds a new directional light.
    /// </summary>
    __host__ DirectionalLight* AddDirectionalLight();

    /// <summary>
    /// Adds a new directional light.
    /// </summary>
    /// <param name="direction">The light's direction.</param>
    __host__ DirectionalLight* AddDirectionalLight( const Vector3& direction );

    /// <summary>
    /// Adds a new directional light.
    /// </summary>
    /// <param name="x">The light's X direction.</param>
    /// <param name="y">The light's Y direction.</param>
    /// <param name="z">The light's Z direction.</param>
    __host__ DirectionalLight* AddDirectionalLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Adds a new point light.
    /// </summary>
    __host__ PointLight* AddPointLight();

    /// <summary>
    /// Adds a new point light.
    /// </summary>
    /// <param name="position">The light's coordinates.</param>
    __host__ PointLight* AddPointLight( const Vector3& position );

    /// <summary>
    /// Adds a new point light.
    /// </summary>
    /// <param name="x">The light's X coordinate.</param>
    /// <param name="y">The light's Y coordinate.</param>
    /// <param name="z">The light's Z coordinate.</param>
    __host__ PointLight* AddPointLight( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets the ambient light's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __host__ void SetAmbientColor( const Color& color );

    /// <summary>
    /// Sets the ambient light's color.
    /// </summary>
    /// <param name="r">The new color's red component.</param>
    /// <param name="g">The new color's green component.</param>
    /// <param name="b">The new color's blue component.</param>
    __host__ void SetAmbientColor( real32 r, real32 g, real32 b );
};

REX_NS_END