#include <rex/Graphics/Geometry/Triangle.hxx>
#include <rex/Graphics/ShadePoint.hxx>

REX_NS_BEGIN

// destroy triangle
__device__ Triangle::~Triangle()
{
}

// get triangle bounds
__device__ BoundingBox Triangle::GetBounds() const
{
    vec3     min    = glm::min( glm::min( _p1, _p2 ), _p3 );
    vec3     max    = glm::max( glm::max( _p1, _p2 ), _p3 );
    BoundingBox bounds = BoundingBox( min, max );
    return bounds;
}

// get triangle normal
__device__ vec3 Triangle::GetNormal() const
{
    vec3 normal = glm::cross( _p2 - _p1, _p3 - _p1 );
    return glm::normalize( normal );
}

// shade-hit triangle
__device__ bool Triangle::Hit( const Ray& ray, real32& tmin, ShadePoint& sp ) const
{
    // adapted from Suffern, 479

    real32 a         = _p1.x - _p2.x;
    real32 b         = _p1.x - _p3.x;
    real32 c         = ray.Direction.x;
    real32 d         = _p1.x - ray.Origin.x;
    real32 e         = _p1.y - _p2.y;
    real32 f         = _p1.y - _p3.y;
    real32 g         = ray.Direction.y;
    real32 h         = _p1.y - ray.Origin.y;
    real32 i         = _p1.z - _p2.z;
    real32 j         = _p1.z - _p3.z;
    real32 k         = ray.Direction.z;
    real32 l         = _p1.z - ray.Origin.z;
    real32 m         = f * k - g * j;
    real32 n         = h * k - g * l;
    real32 p         = f * l - h * j;
    real32 q         = g * i - e * k;
    real32 s         = e * j - f * i;
    real32 invDenom  = 1.0 / ( a * m + b * q + c * s );
    real32 e1        = d * m - b * n - c * p;
    real32 beta      = e1 * invDenom;
    
    if ( beta < real32( 0.0 ) )
    {
        return false;
    }
    
    real32 r     = e * l - h * i;
    real32 e2    = a * n + d * q + c * r;
    real32 gamma = e2 * invDenom;
    
    if ( gamma < real32( 0.0 ) )
    {
        return false;
    }
    
    if ( beta + gamma > real32( 1.0 ) )
    {
        return false;
    }
    
    real32 e3 = a * p - b * r + d * s;
    real32 t  = e3 * invDenom;
    
    if ( t < Math::Epsilon() )
    {
        return false;
    }

    tmin        = t;
    sp.Normal   = GetNormal();
    sp.HitPoint = ray.Origin + t * ray.Direction;
    sp.Material = _material;
    
    return true;
}

// shadow-hit triangle
__device__ bool Triangle::ShadowHit( const Ray& ray, real32& tmin ) const
{
    // adapted from Suffern, 479

    real32 a         = _p1.x - _p2.x;
    real32 b         = _p1.x - _p3.x;
    real32 c         = ray.Direction.x;
    real32 d         = _p1.x - ray.Origin.x;
    real32 e         = _p1.y - _p2.y;
    real32 f         = _p1.y - _p3.y;
    real32 g         = ray.Direction.y;
    real32 h         = _p1.y - ray.Origin.y;
    real32 i         = _p1.z - _p2.z;
    real32 j         = _p1.z - _p3.z;
    real32 k         = ray.Direction.z;
    real32 l         = _p1.z - ray.Origin.z;
    real32 m         = f * k - g * j;
    real32 n         = h * k - g * l;
    real32 p         = f * l - h * j;
    real32 q         = g * i - e * k;
    real32 s         = e * j - f * i;
    real32 invDenom  = 1.0 / ( a * m + b * q + c * s );
    real32 e1        = d * m - b * n - c * p;
    real32 beta      = e1 * invDenom;
    
    if ( beta < real32( 0.0 ) )
    {
        return false;
    }
    
    real32 r     = e * l - h * i;
    real32 e2    = a * n + d * q + c * r;
    real32 gamma = e2 * invDenom;
    
    if ( gamma < real32( 0.0 ) )
    {
        return false;
    }
    
    if ( beta + gamma > real32( 1.0 ) )
    {
        return false;
    }
    
    real32 e3 = a * p - b * r + d * s;
    real32 t  = e3 * invDenom;
    
    if ( t < Math::Epsilon() )
    {
        return false;
    }

    tmin = t;
    return true;
}

REX_NS_END