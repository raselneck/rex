#include <rex/Graphics/Geometry/Sphere.hxx>
#include <rex/Graphics/ShadePoint.hxx>
#include <rex/Math/Math.hxx>
#include <rex/Utility/Logger.hxx>

REX_NS_BEGIN

// destroys this sphere
Sphere::~Sphere()
{
    _radius = 0.0;
}

// get sphere's geometry type
GeometryType Sphere::GetType() const
{
    return GeometryType::Sphere;
}

// gets this sphere's bounds
BoundingBox Sphere::GetBounds() const
{
    Vector3 size( _radius );
    return BoundingBox( _center - size, _center + size );
}

// get the sphere on the GPU
const Geometry* Sphere::GetOnDevice() const
{
    return (const Geometry*)( _dThis );
}

// get this sphere's device material
__device__ const Material* Sphere::GetDeviceMaterial() const
{
    return (const Material*)( _dMaterial );
}

// shade hit this sphere with a ray
__device__ bool Sphere::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // from Suffern, 58

    // this is basically using the quadratic equation solved for x where x = t
    real64  t    = 0.0;
    Vector3 temp = ray.Origin - _center;
    real64  a    = Vector3::Dot( ray.Direction, ray.Direction );
    real64  b    = Vector3::Dot( temp, ray.Direction ) * 2.0;
    real64  c    = Vector3::Dot( temp, temp ) - _radius * _radius;
    real64  disc = b * b - 4.0 * a * c; // discriminant

    // check if the ray misses completely
    if ( disc < 0.0 )
    {
        return false;
    }

    // now we need to check the smaller root (b^2 - 4ac)
    real64 e     = sqrt( disc );
    real64 denom = 1.0 / ( 2.0 * a );
    t = ( -b - e ) * denom;
    if ( t > Math::Epsilon() )
    {
        tmin = t;
        sp.Normal = ( temp + t * ray.Direction ) * _invRadius;
        sp.LocalHitPoint = ray.Origin + t * ray.Direction;

        return true;
    }

    // now we need to check the larger root (b^2 + 4ac)
    t = ( -b + e ) * denom;
    if ( t > Math::Epsilon() )
    {
        tmin = t;
        sp.Normal = ( temp + t * ray.Direction ) * _invRadius;
        sp.LocalHitPoint = ray.Origin + t * ray.Direction;

        return true;
    }

    return false;
}

// shadow hit this sphere with a ray
__device__ bool Sphere::ShadowHit( const Ray& ray, real64& tmin ) const
{
    // this is basically using the quadratic equation solved for x where x = t
    real64  t    = 0.0;
    Vector3 temp = ray.Origin - _center;
    real64  a    = Vector3::Dot( ray.Direction, ray.Direction );
    real64  b    = Vector3::Dot( temp, ray.Direction ) * 2.0;
    real64  c    = Vector3::Dot( temp, temp ) - _radius * _radius;
    real64  disc = b * b - 4.0 * a * c; // discriminant

    // check if the ray misses completely
    if ( disc < 0.0 )
    {
        return false;
    }

    // now we need to check the smaller root (b^2 - 4ac)
    real64 e = sqrt( disc );
    real64 denom = 1.0 / ( 2.0 * a );
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

// handle when the material is changed.
void Sphere::OnChangeMaterial()
{
    // update our "this" pointer
    cudaError_t err = cudaMemcpy( _dThis, this, sizeof( Sphere ), cudaMemcpyHostToDevice );
    if ( err != cudaSuccess )
    {
        REX_DEBUG_LOG( "Failed to update sphere data on device." );
    }
}

REX_NS_END