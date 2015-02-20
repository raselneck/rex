#include "Tracer.hxx"

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
    return Color::Black;
}

// trace the ray
Color Tracer::Trace( const Ray& ray, int depth ) const
{
    return Color::Black;
}

REX_NS_END