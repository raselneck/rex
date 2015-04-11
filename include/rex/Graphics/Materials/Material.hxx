#pragma once

#include "../../Config.hxx"
#include "../Color.hxx"

REX_NS_BEGIN

struct ShadePoint;
class  Light;

/// <summary>
/// An enumeration of possible material types.
/// </summary>
enum class MaterialType
{
    Matte,
    Phong
};

/// <summary>
/// Defines the base for all materials.
/// </summary>
class Material
{
    REX_NONCOPYABLE_CLASS( Material );

protected:
    friend class Geometry;

    void* _dThis;

    /// <summary>
    /// Copies this material for geometry.
    /// </summary>
    __host__ virtual Material* Copy() const = 0;

public:
    /// <summary>
    /// Creates a new material.
    /// </summary>
    __host__ Material();

    /// <summary>
    /// Destroys this material.
    /// </summary>
    __host__ virtual ~Material();

    /// <summary>
    /// Gets this material on the device.
    /// </summary>
    __host__ virtual const Material* GetOnDevice() const = 0;

    /// <summary>
    /// Gets this material's type.
    /// </summary>
    __both__ virtual MaterialType GetType() const = 0;

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    /// <param name="lights">All of the lights in the scene.</param>
    /// <param name="lightCount">The number of lights in the scene</param>
    __device__ virtual Color Shade( ShadePoint& sp, const Light** lights, uint32 lightCount ) const;
};

REX_NS_END