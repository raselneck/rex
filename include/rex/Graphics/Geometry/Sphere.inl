REX_NS_BEGIN

// create new sphere
template<typename T> __device__ Sphere::Sphere( const T& material )
    : Geometry( GeometryType::Sphere, material ),
      _radius ( 0.0 )
{
}

// create new sphere
template<typename T> __device__ Sphere::Sphere( const T& material, const Vector3& center, real_t radius )
    : Geometry( GeometryType::Sphere, material ),
      _center ( center ),
      _radius ( radius )
{
}

REX_NS_END