#include "RayCastTracer.hxx"
#include <rex/Scene/Scene.hxx>

REX_NS_BEGIN

// create tracer
RayCastTracer::RayCastTracer( Scene* scene )
    : Tracer( scene ), _sp( scene )
{
}

// destroy tracer
RayCastTracer::~RayCastTracer()
{
}

// trace a ray
Color RayCastTracer::Trace( const rex::Ray& ray ) const
{
    return Trace( ray, 0 );
}

// trace a ray based on depth
Color RayCastTracer::Trace( const rex::Ray& ray, int32 depth ) const
{
    _sp.Reset();

    _scene->HitObjects( ray, _sp );
    if ( _sp.HasHit )
    {
        _sp.Ray = ray;
        return _sp.Material->Shade( _sp );
    }

    return _scene->GetBackgroundColor();
}

REX_NS_END