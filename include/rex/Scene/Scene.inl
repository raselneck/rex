#ifndef __REX_SCENE_INL
#define __REX_SCENE_INL

#include "Scene.hxx"

REX_NS_BEGIN

// add plane
template<class T> inline Handle<Plane> Scene::AddPlane( const Vector3& point, const Vector3& normal, const T& material )
{
    Handle<Plane> plane( new Plane( point, normal ) );
    plane->SetMaterial( material );
    _objects.push_back( plane );
    return plane;
}

// add sphere
template<class T> inline Handle<Sphere> Scene::AddSphere( const Vector3& center, real64 radius, const T& material )
{
    Handle<Sphere> sphere( new Sphere( center, radius ) );
    sphere->SetMaterial( material );
    _objects.push_back( sphere );
    return sphere;
}

// set camera type
template<class T> inline void Scene::SetCameraType()
{
    _camera.reset( new T() );
}

// set sampler type
template<class T> inline void Scene::SetSamplerType()
{
    SetSamplerType<T>( REX_DEFAULT_SAMPLES, REX_DEFAULT_SETS );
}

// set sampler type w/ sample count
template<class T> inline void Scene::SetSamplerType( int32 samples )
{
    SetSamplerType<T>( samples, REX_DEFAULT_SETS );
}

// set sampler type w/ sample count, set count
template<class T> inline void Scene::SetSamplerType( int32 samples, int32 sets )
{
    _sampler.reset( new T( samples, sets ) );
    _sampler->GenerateSamples();
}

// set tracer type
template<class T> inline void Scene::SetTracerType()
{
    _tracer.reset( new T( this ) );
}

REX_NS_END

#endif