#include <rex/Tracers/Tracer.hxx>

REX_NS_BEGIN

// new ray tracer
Tracer::Tracer( Scene* scene )
    : _scene( scene )
{
}

// destroy ray tracer
Tracer::~Tracer()
{
    const_cast<Scene*>( _scene ) = 0;
}

// trace the ray
Color Tracer::Trace( const Ray& ray ) const
{
    return Color::Magenta;
}

// trace the ray w/ depth count
Color Tracer::Trace( const Ray& ray, int depth ) const
{
    return Color::Magenta;
}

// trace the ray w/ depth count, min object distance
Color Tracer::Trace( const Ray& ray, int32 depth, real32& tmin ) const
{
    return Color::Magenta;
}

REX_NS_END