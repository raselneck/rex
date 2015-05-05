#pragma once

#include "Material.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an emissive material.
/// </summary>
class EmissiveMaterial : public Material
{
protected:
    friend class Geometry;

    Color  _color;
    real32 _radianceScale;

    /// <summary>
    /// Copies this material for geometry.
    /// </summary>
    __device__ virtual Material* Copy() const;

public:
    /// <summary>
    /// Creates a new emissive material.
    /// </summary>
    __device__ EmissiveMaterial();

    /// <summary>
    /// Creates a new emissive material.
    /// </summary>
    /// <param name="color">The material's color.</param>
    /// <param name="ls">The material's radiance scale.</param>
    __device__ EmissiveMaterial( const Color& color, real32 ls );

    /// <summary>
    /// Destroys this emissive material.
    /// </summary>
    __device__ virtual ~EmissiveMaterial();

    /// <summary>
    /// Gets an area light shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    /// <param name="lights">All of the lights in the current scene.</param>
    /// <param name="octree">The octree containing the objects to pass to the lights.</param>
    __device__ virtual Color AreaLightShade( ShadePoint& sp ) const;

    /// <summary>
    /// Gets this material's color.
    /// </summary>
    __device__ const Color& GetColor() const;

    /// <summary>
    /// Gets this material's radiance scale.
    /// </summary>
    __device__ real32 GetRadianceScale() const;

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    /// <param name="lights">All of the lights in the current scene.</param>
    /// <param name="octree">The octree containing the objects to pass to the lights.</param>
    __device__ virtual Color Shade( ShadePoint& sp ) const;

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __device__ void SetColor( const Color& color );

    /// <summary>
    /// Sets this material's radiance scale.
    /// </summary>
    /// <param name="ls">The new radiance scale.</param>
    __device__ void SetRadianceScale( real32 ls );
};

REX_NS_END