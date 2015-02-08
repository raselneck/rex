#include "Ray.hxx"

REX_NS_BEGIN

// new ray
Ray::Ray()
{
}

// new ray
Ray::Ray( const Vector3& origin, const Vector3& direction )
    : Origin( origin ),
      Direction( direction )
{
}

// destroy ray
Ray::~Ray()
{
}

REX_NS_END