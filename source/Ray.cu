#include <rex/Math/Ray.hxx>

REX_NS_BEGIN

// new ray
Ray::Ray()
{
}

// new ray
Ray::Ray( const vec3& origin, const vec3& direction )
    : Origin( origin ),
      Direction( direction )
{
}

// destroy ray
Ray::~Ray()
{
}

REX_NS_END