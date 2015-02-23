#ifndef __REX_TRACER_HXX
#define __REX_TRACER_HXX

#include "../Config.hxx"
#include "../Utility/Color.hxx"
#include "../Utility/Ray.hxx"

REX_NS_BEGIN

class Scene;

/// <summary>
/// Defines the base for all ray tracers.
/// </summary>
class Tracer
{
protected:
    Scene* const _scene;

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

    /// <summary>
    /// Traces the given ray and returns the color it generates.
    /// </summary>
    /// <param name="ray">The ray to trace.</param>
    /// <param name="depth">The current ray depth.</param>
    virtual Color Trace( const Ray& ray, int32 depth ) const;

    /// <summary>
    /// Traces the given ray and returns the color it generates.
    /// </summary>
    /// <param name="ray">The ray to trace.</param>
    /// <param name="depth">The current ray depth.</param>
    /// <param name="tmin">The current minimum object distance.</param>
    virtual Color Trace( const Ray& ray, int32 depth, real32& tmin ) const;
};

REX_NS_END

#endif