#pragma once

#include "../../Config.hxx"
#include "../../CUDA/DeviceList.hxx"
#include "../Geometry/Octree.hxx"
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
    REX_NONCOPYABLE_CLASS( Material )
    REX_IMPLEMENT_DEVICE_MEM_OPS()

protected:
    friend class Geometry;

    MaterialType _type;

    /// <summary>
    /// Copies this material for geometry.
    /// </summary>
    __device__ virtual Material* Copy() const = 0;

public:
    /// <summary>
    /// Creates a new material.
    /// </summary>
    /// <param name="type">The material type.</param>
    __device__ Material( MaterialType type );

    /// <summary>
    /// Destroys this material.
    /// </summary>
    __device__ virtual ~Material();

    /// <summary>
    /// Gets this material's type.
    /// </summary>
    __device__ MaterialType GetType() const;

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    /// <param name="lights">All of the lights in the current scene.</param>
    /// <param name="octree">The octree containing the objects to pass to the lights.</param>
    __device__ virtual Color Shade( ShadePoint& sp, const DeviceList<Light*>* lights, const Octree* octree ) const;
};

REX_NS_END