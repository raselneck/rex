#include "RayCastTracer.hxx"
#include <rex/Scene/Scene.hxx>

REX_NS_BEGIN

// create tracer
RayCastTracer::RayCastTracer( Scene* scene )
    : Tracer( scene )
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
    ShadePoint sp( _scene );

    _scene->HitObjects( ray, sp );
    if ( sp.HasHit )
    {
        sp.Ray = ray;
        return sp.Material->Shade( sp );
    }

    return _scene->GetBackgroundColor();
}

REX_NS_END