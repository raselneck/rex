#ifndef __REX_MESH_HXX
#define __REX_MESH_HXX

#include "Geometry.hxx"
#include "Octree.hxx"
#include "Triangle.hxx"
#include <vector>

struct aiScene;
struct aiMesh;
struct aiNode;

REX_NS_BEGIN

/// <summary>
/// Defines a mesh.
/// </summary>
class Mesh : public Geometry
{
    mutable std::vector<const Geometry*> _queryObjects;
    std::vector<Handle<Triangle>> _triangles;
    Handle<Octree> _octree;
    Vector3 _center;
    String _name;

    /// <summary>
    /// Processes an Assimp node.
    /// </summary>
    /// <param name="node">The current node to process.</param>
    /// <param name="scene">The scene to process.</param>
    static void ProcessAssimpNode( aiNode* node, const aiScene* scene, std::vector<Handle<Mesh>>& meshes );

    /// <summary>
    /// Processes an Assimp mesh.
    /// </summary>
    /// <param name="aMesh">The Assimp mesh to process.</param>
    /// <param name="scene">The Assimp scene to use.</param>
    /// <param name="rMesh">The Rex mesh to populate.</param>
    static void ProcessAssimpMesh( aiMesh* aMesh, const aiScene* scene, Handle<Mesh>& rMesh );

    /// <summary>
    /// Builds the octree for this mesh.
    /// </summary>
    void BuildOctree();

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
    /// Gets the center of this mesh.
    /// </summary>
    const Vector3& GetCenter() const;

    /// <summary>
    /// Gets this mesh's name.
    /// </summary>
    const String& GetName() const;

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
    /// Moves this mesh.
    /// </summary>
    /// <param name="trans">The translation.</param>
    void Move( const Vector3& trans );

    /// <summary>
    /// Moves this mesh.
    /// </summary>
    /// <param name="x">The X translation.</param>
    /// <param name="y">The Y translation.</param>
    /// <param name="z">The Z translation.</param>
    void Move( real64 x, real64 y, real64 z );

    /// <summary>
    /// Sets the material for this mesh.
    /// </summary>
    /// <param name="material">The existing material.</param>
    virtual void SetMaterial( const Handle<Material>& material );

    /// <summary>
    /// Sets the material for this mesh.
    /// </summary>
    /// <param name="material">The material.</param>
    virtual void SetMaterial( const Material& material );

    /// <summary>
    /// Attempts to load all of the meshes found within the given file.
    /// </summary>
    /// <param name="fname">The file name.</param>
    /// <param name="meshes">The mesh collection to populate.</param>
    static bool LoadFile( const String& fname, std::vector<Handle<Mesh>>& meshes );
};

REX_NS_END

#endif