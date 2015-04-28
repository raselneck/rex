REX_NS_BEGIN

// create new sphere
template<typename T> __device__ Sphere::Sphere( const T& material )
    : Geometry( GeometryType::Sphere, material ),
      _radius ( 0.0 )
{
}

// create new sphere
template<typename T> __device__ Sphere::Sphere( const T& material, const vec3& center, real32 radius )
    : Geometry( GeometryType::Sphere, material ),
      _center ( center ),
      _radius ( radius )
{
}

REX_NS_END