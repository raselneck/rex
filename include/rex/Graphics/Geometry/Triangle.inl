REX_NS_BEGIN

// create a new triangle
template<typename T> __device__ Triangle::Triangle( const T& material )
    : Geometry( GeometryType::Triangle, material )
{
}

// create a new triangle
template<typename T> __device__ Triangle::Triangle( const T& material, const vec3& p1, const vec3& p2, const vec3& p3 )
    : Geometry( GeometryType::Triangle, material ),
      _p1( p1 ),
      _p2( p2 ),
      _p3( p3 )
{
}

REX_NS_END