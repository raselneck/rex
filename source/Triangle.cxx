#include <rex/Geometry/Triangle.hxx>
#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

// create triangle
Triangle::Triangle()
{
}

// destroy triangle
Triangle::~Triangle()
{
}

// get triangle bounds
BoundingBox Triangle::GetBounds() const
{
    Vector3 min = Vector3::Min( Vector3::Min( P1, P2 ), P3 );
    Vector3 max = Vector3::Max( Vector3::Max( P1, P2 ), P3 );

    return BoundingBox( min, max );
}

// get triangle normal
Vector3 Triangle::GetNormal() const
{
    Vector3 normal = Vector3::Cross( P2 - P1, P3 - P1 );
    return Vector3::Normalize( normal );
}

// get geometry type
GeometryType Triangle::GetType() const
{
    return GeometryType::Triangle;
}

// ray hit triangle
bool Triangle::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // adapted from Suffern, 479

    real64 a         = P1.X - P2.X,
           b         = P1.X - P3.X,
           c         = ray.Direction.X,
           d         = P1.X - ray.Origin.X; 
    real64 e         = P1.Y - P2.Y,
           f         = P1.Y - P3.Y,
           g         = ray.Direction.Y,
           h         = P1.Y - ray.Origin.Y;
    real64 i         = P1.Z - P2.Z,
           j         = P1.Z - P3.Z,
           k         = ray.Direction.Z,
           l         = P1.Z - ray.Origin.Z;
    real64 m         = f * k - g * j,
           n         = h * k - g * l,
           p         = f * l - h * j;
    real64 q         = g * i - e * k,
           s         = e * j - f * i;
    real64 invDenom  = 1.0 / (a * m + b * q + c * s);
    real64 e1        = d * m - b * n - c * p;
    real64 beta      = e1 * invDenom;
    
    if ( beta < 0.0 )
    {
        return false;
    }
    
    real64 r = e * l - h * i;
    real64 e2 = a * n + d * q + c * r;
    real64 gamma = e2 * invDenom;
    
    if ( gamma < 0.0 )
    {
        return false;
    }
    
    if ( beta + gamma > 1.0 )
    {
        return false;
    }
    
    real64 e3 = a * p - b * r + d * s;
    real64 t = e3 * invDenom;
    
    if ( t < Math::EPSILON )
    {
        return false;
    }

    tmin                = t;
    sp.Normal           = GetNormal();
    sp.LocalHitPoint    = ray.Origin + t * ray.Direction;
    
    return true;
}

// shadow hit triangle
bool Triangle::ShadowHit( const Ray& ray, real64& tmin ) const
{
    real64 a         = P1.X - P2.X,
           b         = P1.X - P3.X,
           c         = ray.Direction.X,
           d         = P1.X - ray.Origin.X; 
    real64 e         = P1.Y - P2.Y,
           f         = P1.Y - P3.Y,
           g         = ray.Direction.Y,
           h         = P1.Y - ray.Origin.Y;
    real64 i         = P1.Z - P2.Z,
           j         = P1.Z - P3.Z,
           k         = ray.Direction.Z,
           l         = P1.Z - ray.Origin.Z;
    real64 m         = f * k - g * j,
           n         = h * k - g * l,
           p         = f * l - h * j;
    real64 q         = g * i - e * k,
           s         = e * j - f * i;
    real64 invDenom  = 1.0 / (a * m + b * q + c * s);
    real64 e1        = d * m - b * n - c * p;
    real64 beta      = e1 * invDenom;
    
    if ( beta < 0.0 )
    {
        return false;
    }
    
    real64 r = e * l - h * i;
    real64 e2 = a * n + d * q + c * r;
    real64 gamma = e2 * invDenom;
    
    if ( gamma < 0.0 )
    {
        return false;
    }
    
    if ( beta + gamma > 1.0 )
    {
        return false;
    }
    
    real64 e3 = a * p - b * r + d * s;
    real64 t = e3 * invDenom;
    
    if ( t < Math::EPSILON )
    {
        return false;
    }

    tmin                = t;

    return true;
}

REX_NS_END