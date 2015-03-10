#ifndef __REX_MESH_HXX
#define __REX_MESH_HXX

#include "Geometry.hxx"
#include "Octree.hxx"
#include "Triangle.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines a mesh.
/// </summary>
class Mesh : public Geometry
{
    mutable std::vector<const Geometry*> _queryObjects;
    std::vector<Handle<Triangle>> _triangles;
    Handle<Octree> _octree;

public:
    /// <summary>
    /// Creates a new mesh.
    /// </summary>
    Mesh();

    /// <summary>
    /// Destroys this mesh.
    /// </summary>
    virtual ~Mesh();

    /// <summary>
    /// Gets this mesh's geometry type.
    /// </summary>
    virtual GeometryType GetType() const;

    /// <summary>
    /// Gets this mesh's bounds.
    /// </summary>
    virtual BoundingBox GetBounds() const;

    /// <summary>
    /// Checks to see if the given ray hits this mesh. If it does, the shading
    /// point information is populated and the collision distance is recorded.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    /// <param name="sp">The shading point information.</param>
    virtual bool Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const;

    /// <summary>
    /// Performs the same thing as a normal ray hit, but for shadow rays.
    /// </summary>
    /// <param name="ray">The ray to check.</param>
    /// <param name="tmin">The distance to intersection.</param>
    virtual bool ShadowHit( const Ray& ray, real64& tmin ) const;

    /// <summary>
    /// Attempts to load the given file as a model.
    /// </summary>
    /// <param name="fname">The file name.</param>
    bool Load( const String& fname );

    /// <summary>
    /// Sets the material for this mesh.
    /// </summary>
    /// <param name="material">The existing material.</param>
    template<class T> virtual void SetMaterial( const Handle<T>& material );

    /// <summary>
    /// Sets the material for this mesh.
    /// </summary>
    /// <param name="material">The material.</param>
    template<class T> virtual void SetMaterial( const T& material );
};

REX_NS_END

#include "Mesh.inl"
#endif