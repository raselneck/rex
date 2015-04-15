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
    Vector3     min    = Vector3::Min( Vector3::Min( _p1, _p2 ), _p3 );
    Vector3     max    = Vector3::Max( Vector3::Max( _p1, _p2 ), _p3 );
    BoundingBox bounds = BoundingBox( min, max );
    return bounds;
}

// get triangle normal
__device__ Vector3 Triangle::GetNormal() const
{
    Vector3 normal = Vector3::Cross( _p2 - _p1, _p3 - _p1 );
    return Vector3::Normalize( normal );
}

// shade-hit triangle
__device__ bool Triangle::Hit( const Ray& ray, real_t& tmin, ShadePoint& sp ) const
{
    // adapted from Suffern, 479

    real_t a         = _p1.X - _p2.X;
    real_t b         = _p1.X - _p3.X;
    real_t c         = ray.Direction.X;
    real_t d         = _p1.X - ray.Origin.X;
    real_t e         = _p1.Y - _p2.Y;
    real_t f         = _p1.Y - _p3.Y;
    real_t g         = ray.Direction.Y;
    real_t h         = _p1.Y - ray.Origin.Y;
    real_t i         = _p1.Z - _p2.Z;
    real_t j         = _p1.Z - _p3.Z;
    real_t k         = ray.Direction.Z;
    real_t l         = _p1.Z - ray.Origin.Z;
    real_t m         = f * k - g * j;
    real_t n         = h * k - g * l;
    real_t p         = f * l - h * j;
    real_t q         = g * i - e * k;
    real_t s         = e * j - f * i;
    real_t invDenom  = 1.0 / ( a * m + b * q + c * s );
    real_t e1        = d * m - b * n - c * p;
    real_t beta      = e1 * invDenom;
    
    if ( beta < real_t( 0.0 ) )
    {
        return false;
    }
    
    real_t r     = e * l - h * i;
    real_t e2    = a * n + d * q + c * r;
    real_t gamma = e2 * invDenom;
    
    if ( gamma < real_t( 0.0 ) )
    {
        return false;
    }
    
    if ( beta + gamma > real_t( 1.0 ) )
    {
        return false;
    }
    
    real_t e3 = a * p - b * r + d * s;
    real_t t  = e3 * invDenom;
    
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
__device__ bool Triangle::ShadowHit( const Ray& ray, real_t& tmin ) const
{
    // adapted from Suffern, 479

    real_t a         = _p1.X - _p2.X;
    real_t b         = _p1.X - _p3.X;
    real_t c         = ray.Direction.X;
    real_t d         = _p1.X - ray.Origin.X;
    real_t e         = _p1.Y - _p2.Y;
    real_t f         = _p1.Y - _p3.Y;
    real_t g         = ray.Direction.Y;
    real_t h         = _p1.Y - ray.Origin.Y;
    real_t i         = _p1.Z - _p2.Z;
    real_t j         = _p1.Z - _p3.Z;
    real_t k         = ray.Direction.Z;
    real_t l         = _p1.Z - ray.Origin.Z;
    real_t m         = f * k - g * j;
    real_t n         = h * k - g * l;
    real_t p         = f * l - h * j;
    real_t q         = g * i - e * k;
    real_t s         = e * j - f * i;
    real_t invDenom  = 1.0 / ( a * m + b * q + c * s );
    real_t e1        = d * m - b * n - c * p;
    real_t beta      = e1 * invDenom;
    
    if ( beta < real_t( 0.0 ) )
    {
        return false;
    }
    
    real_t r     = e * l - h * i;
    real_t e2    = a * n + d * q + c * r;
    real_t gamma = e2 * invDenom;
    
    if ( gamma < real_t( 0.0 ) )
    {
        return false;
    }
    
    if ( beta + gamma > real_t( 1.0 ) )
    {
        return false;
    }
    
    real_t e3 = a * p - b * r + d * s;
    real_t t  = e3 * invDenom;
    
    if ( t < Math::Epsilon() )
    {
        return false;
    }

    tmin = t;
    return true;
}

REX_NS_END