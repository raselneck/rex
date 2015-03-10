#ifndef __REX_HXX
#define __REX_HXX

/**
 * POSSIBLY TODO
 * 1) Ch. 10 explains depth-of-field with a thin lens camera
 * 2) Ch. 11 explains nonlinear projections with both a fish-eye and spherical panoramic camera
 * 3) Ch. 12 explains stereoscopy with a stereoscopic camera (for use with an Oculus Rift)
 */

#include "BRDFs/BRDF.hxx"
#include "BRDFs/GlossySpecularBRDF.hxx"
#include "BRDFs/LambertianBRDF.hxx"
#include "Cameras/Camera.hxx"
#include "Cameras/PerspectiveCamera.hxx"
#include "Geometry/BoundingBox.hxx"
#include "Geometry/Geometry.hxx"
#include "Geometry/Octree.hxx"
#include "Geometry/Mesh.hxx"
#include "Geometry/Plane.hxx"
#include "Geometry/Sphere.hxx"
#include "Geometry/Triangle.hxx"
#include "Lights/AmbientLight.hxx"
#include "Lights/DirectionalLight.hxx"
#include "Lights/Light.hxx"
#include "Lights/PointLight.hxx"
#include "Materials/Material.hxx"
#include "Materials/MatteMaterial.hxx"
#include "Materials/PhongMaterial.hxx"
#include "Samplers/HammersleySampler.hxx"
#include "Samplers/JitteredSampler.hxx"
#include "Samplers/MultiJitteredSampler.hxx"
#include "Samplers/NRooksSampler.hxx"
#include "Samplers/RegularSampler.hxx"
#include "Samplers/Sampler.hxx"
#include "Scene/Scene.hxx"
#include "Scene/ShadePoint.hxx"
#include "Scene/ViewPlane.hxx"
#include "Tracers/Tracer.hxx"
#include "Tracers/RayCastTracer.hxx"
#include "Utility/Color.hxx"
#include "Utility/Image.hxx"
#include "Utility/Math.hxx"
#include "Utility/Matrix.hxx"
#include "Utility/Random.hxx"
#include "Utility/Ray.hxx"
#include "Utility/Timer.hxx"
#include "Utility/Vector2.hxx"
#include "Utility/Vector3.hxx"

#endif