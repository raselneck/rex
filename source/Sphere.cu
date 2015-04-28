#include <rex/Graphics/Geometry/Sphere.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Math/Math.hxx>
#include <rex/Utility/Logger.hxx>

REX_NS_BEGIN

// destroys this sphere
__device__ Sphere::~Sphere()
{
    _radius = 0.0;
}

// get bounds
__device__ BoundingBox Sphere::GetBounds() const
{
    vec3     size   = vec3( _radius );
    BoundingBox bounds = BoundingBox( _center - size, _center + size );
    return bounds;
}

// shade hit this sphere with a ray
__device__ bool Sphere::Hit( const Ray& ray, real32& tmin, ShadePoint& sp ) const
{
    // from Suffern, 58

    // this is basically using the quadratic equation solved for x where x = t
    real32  t    = 0.0;
    vec3 temp = ray.Origin - _center;
    real32  a    = glm::dot( ray.Direction, ray.Direction );
    real32  b    = glm::dot( temp, ray.Direction ) * 2.0;
    real32  c    = glm::dot( temp, temp ) - _radius * _radius;
    real32  disc = b * b - 4.0 * a * c; // discriminant

    // check if the ray misses completely
    if ( disc < 0.0 )
    {
        return false;
    }

    // now we need to check the smaller root (b^2 - 4ac)
    real32 e     = sqrt( disc );
    real32 denom = 1.0 / ( 2.0 * a );
    t = ( -b - e ) * denom;
    if ( t > Math::Epsilon() )
    {
        tmin        = t;
        sp.Normal   = ( temp + t * ray.Direction ) / _radius;
        sp.HitPoint = ray.Origin + t * ray.Direction;
        sp.Material = _material;

        return true;
    }

    // now we need to check the larger root (b^2 + 4ac)
    t = ( -b + e ) * denom;
    if ( t > Math::Epsilon() )
    {
        tmin        = t;
        sp.Normal   = ( temp + t * ray.Direction ) / _radius;
        sp.HitPoint = ray.Origin + t * ray.Direction;
        sp.Material = _material;

        return true;
    }

    return false;
}

// shadow hit this sphere with a ray
__device__ bool Sphere::ShadowHit( const Ray& ray, real32& tmin ) const
{
    // this is basically using the quadratic equation solved for x where x = t
    real32  t    = 0.0;
    vec3 temp = ray.Origin - _center;
    real32  a    = glm::dot( ray.Direction, ray.Direction );
    real32  b    = glm::dot( temp, ray.Direction ) * 2.0;
    real32  c    = glm::dot( temp, temp ) - _radius * _radius;
    real32  disc = b * b - 4.0 * a * c; // discriminant

    // check if the ray misses completely
    if ( disc < 0.0 )
    {
        return false;
    }

    // now we need to check the smaller root (b^2 - 4ac)
    real32 e     = sqrt( disc );
    real32 denom = 1.0 / ( 2.0 * a );
    t = ( -b - e ) * denom;
    if ( t > Math::Epsilon() )
    {
        tmin = t;
        return true;
    }

    // now we need to check the larger root (b^2 + 4ac)
    t = ( -b + e ) * denom;
    if ( t > Math::Epsilon() )
    {
        tmin = t;
        return true;
    }

    return false;
}

REX_NS_END