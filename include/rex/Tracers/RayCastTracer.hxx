#ifndef __RAYCASTTRACER_HXX
#define __RAYCASTTRACER_HXX

#include "Tracer.hxx"
#include "../Scene/ShadePoint.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a basic ray cast tracer.
/// </summary>
class RayCastTracer : public Tracer
{
public:
    /// <summary>
    /// Creates a new ray cast tracer.
    /// </summary>
    /// <param name="scene">The scene to trace for.</param>
    RayCastTracer( Scene* scene );

    /// <summary>
    /// Destroys this ray cast tracer.
    /// </summary>
    virtual ~RayCastTracer();

    /// <summary>
    /// Traces a ray.
    /// </summary>
    /// <param name="ray">The ray to trace.</param>
    virtual rex::Color Trace( const rex::Ray& ray ) const;

    /// <summary>
    /// Traces a ray based on the current recursive depth.
    /// </summary>
    /// <param name="ray">The ray to trace.</param>
    /// <param name="depth">The current recursive depth.</param>
    virtual rex::Color Trace( const rex::Ray& ray, int32 depth ) const;
};

REX_NS_END

#endif