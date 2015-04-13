#pragma once

#include "Material.hxx"
#include "../BRDFs/LambertianBRDF.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a matte material.
/// </summary>
class MatteMaterial : public Material
{
    REX_IMPLEMENT_DEVICE_MEM_OPS()

protected:
    friend class Geometry;

    LambertianBRDF _ambient;
    LambertianBRDF _diffuse;

    /// <summary>
    /// Copies this material for geometry.
    /// </summary>
    __device__ virtual Material* Copy() const;

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    /// <param name="type">The actual material type.</param>
    __device__ MatteMaterial( const Color& color, real_t ka, real_t kd, MaterialType type );

public:
    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    __device__ MatteMaterial();

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    __device__ MatteMaterial( const Color& color );

    /// <summary>
    /// Creates a new matte material.
    /// </summary>
    /// <param name="color">The initial material color.</param>
    /// <param name="ka">The initial ambient coefficient.</param>
    /// <param name="kd">The initial diffuse coefficient.</param>
    __device__ MatteMaterial( const Color& color, real_t ka, real_t kd );

    /// <summary>
    /// Destroys this matte material.
    /// </summary>
    __device__ virtual ~MatteMaterial();

    /// <summary>
    /// Gets the ambient BRDF's diffuse coefficient.
    /// </summary>
    __device__ real_t GetAmbientCoefficient() const;

    /// <summary>
    /// Gets this material's color.
    /// </summary>
    __device__ Color GetColor() const;

    /// <summary>
    /// Gets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    __device__ real_t GetDiffuseCoefficient() const;
    
    /// <summary>
    /// Sets the ambient BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="ka">The new ambient coefficient.</param>
    __device__ virtual void SetAmbientCoefficient( real_t ka );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="color">The new color.</param>
    __device__ virtual void SetColor( const Color& color );

    /// <summary>
    /// Sets this material's color.
    /// </summary>
    /// <param name="r">The new color's red component..</param>
    /// <param name="g">The new color's green component..</param>
    /// <param name="b">The new color's blue component..</param>
    __device__ virtual void SetColor( real_t r, real_t g, real_t b );

    /// <summary>
    /// Sets the diffuse BRDF's diffuse coefficient.
    /// </summary>
    /// <param name="kd">The new diffuse coefficient.</param>
    __device__ virtual void SetDiffuseCoefficient( real_t kd );

    /// <summary>
    /// Gets a shaded color given hit point data.
    /// </summary>
    /// <param name="sp">The hit point data.</param>
    /// <param name="lights">All of the lights in the current scene.</param>
    /// <param name="octree">The octree containing the objects to pass to the lights.</param>
    __device__ virtual Color Shade( ShadePoint& sp, const DeviceList<Light*>* lights, const Octree* octree ) const;
};

REX_NS_END