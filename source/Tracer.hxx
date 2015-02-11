#ifndef __REX_TRACER_HXX
#define __REX_TRACER_HXX
#pragma once

#include "Config.hxx"
#include "Color.hxx"
#include "Ray.hxx"

REX_NS_BEGIN

class Scene;

/// <summary>
/// Defines the base for all ray tracers.
/// </summary>
class Tracer
{
protected:
    Scene* const _pScene;

    /// <summary>
    /// Traces the given ray and returns the color it generates.
    /// </summary>
    /// <param name="ray">The ray to trace.</param>
    /// <param name="depth">The current ray depth.</param>
    virtual Color Trace( const Ray& ray, int depth ) const;

public:
    /// <summary>
    /// Creates a new ray tracer.
    /// </summary>
    /// <param name="scene">The scene to be traced.</param>
    Tracer( Scene* scene );

    /// <summary>
    /// Destroys this ray tracer.
    /// </summary>
    virtual ~Tracer();

    /// <summary>
    /// Traces the given ray and returns the color it generates.
    /// </summary>
    /// <param name="ray">The ray to trace.</param>
    virtual Color Trace( const Ray& ray ) const;
};

REX_NS_END

#endif