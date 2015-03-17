#include <rex/Geometry/Mesh.hxx>
#include <rex/Scene/ShadePoint.hxx>
#include <rex/Debug.hxx>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/ProgressHandler.hpp>
#include <iostream>

REX_NS_BEGIN

// converts an Assimp vector to our vector type
Vector3 AIVectorToRexVector( const aiVector3D& ai )
{
    return Vector3( ai.x,
                    ai.y,
                    ai.z );
}

// create mesh
Mesh::Mesh()
{
}

// destroy mesh
Mesh::~Mesh()
{
}

// get the center of the mesh
const Vector3& Mesh::GetCenter() const
{
    return _center;
}

// get the mesh's name
const String& Mesh::GetName() const
{
    return _name;
}

// get mesh triangle count
uint64 Mesh::GetTriangleCount() const
{
    return static_cast<uint64>( _triangles.size() );
}

// get mesh geometry type
GeometryType Mesh::GetType() const
{
    return GeometryType::Mesh;
}

// get mesh bounds
BoundingBox Mesh::GetBounds() const
{
    Vector3 min;
    Vector3 max;

    if ( _octree )
    {
        min = _octree->GetBounds().GetMin();
        max = _octree->GetBounds().GetMax();

        min -= Vector3( 1.0 );
        max += Vector3( 1.0 );
    }

    return BoundingBox( min, max );
}

// hit mesh
bool Mesh::Hit( const Ray& ray, real64& tmin, ShadePoint& sp ) const
{
    // NOTE : This is literally the same as the scene's HitObjects method


    // prepare to check objects
    real64  t = 0.0;
            tmin = Math::HUGE_VALUE;
    Vector3 normal;
    Vector3 localHitPoint;


    // only get the objects that the ray hits
    _octree->QueryIntersections( ray, _queryObjects );


    // iterate through the hit objects
    for ( auto& obj : _queryObjects )
    {
        if ( obj->Hit( ray, t, sp ) && ( t < tmin ) )
        {
            sp.HasHit = true;
            sp.Material = const_cast<Material*>( obj->GetMaterial() ); // TODO : BAD!
            sp.HitPoint = ray.Origin + t * ray.Direction;

            tmin = t;
            normal = sp.Normal;
            localHitPoint = sp.LocalHitPoint;
        }
    }


    // restore hit point data from closest object
    if ( sp.HasHit )
    {
        sp.T = tmin;
        sp.Normal = normal;
        sp.LocalHitPoint = localHitPoint;
    }

    return sp.HasHit;
}

// shadow hit mesh
bool Mesh::ShadowHit( const Ray& ray, real64& tmin ) const
{
    // TODO : Does this really need to check through every single object
    // that the ray has hit after querying the octree?
    // If the size of the _queryObjects collection is > 0 then the ray
    // has hit, and we only need the minimum hit distance.

    // NOTE : You guessed it! This is basically the same as the scene's
    // ShadowHitObjects method.


    real64 t = 0.0;

    // query which objects the ray hits
    _octree->QueryIntersections( ray, _queryObjects );

    // only hit the objects the ray actually hits
    bool hasHit = false;
    for ( auto& obj : _queryObjects )
    {
        if ( obj->ShadowHit( ray, t ) && ( t < tmin ) )
        {
            hasHit = true;
            tmin = t;
        }
    }

    return hasHit;
}

// build octree data for the mesh
void Mesh::BuildOctree()
{
    // first we need to get the bounds for the octree
    Vector3 bMin(  Math::HUGE_VALUE );
    Vector3 bMax( -Math::HUGE_VALUE );
    for ( auto& tri : _triangles )
    {
        bMin = Vector3::Min( bMin, Vector3::Min( tri->P1, Vector3::Min( tri->P2, tri->P3 ) ) );
        bMax = Vector3::Max( bMax, Vector3::Max( tri->P1, Vector3::Max( tri->P2, tri->P3 ) ) );
    }

    // now create (or reset) the octree
    const uint32 maxItemCount = static_cast<uint32>( Math::Max( 4ULL, _triangles.size() / 12 ) );
    _octree.reset( new Octree( bMin, bMax, maxItemCount ) );


    // aaaanndd add all of the triangles to the octree :)
    for ( auto& tri : _triangles )
    {
        _octree->Add( tri.get() );
    }
}

// move mesh
void Mesh::Move( const Vector3& trans )
{
    _center += trans;

    // move the triangles
    for ( auto& tri : _triangles )
    {
        tri->P1 += trans;
        tri->P2 += trans;
        tri->P3 += trans;
    }

    // re-build the octree
    BuildOctree();
}

// move mesh
void Mesh::Move( real64 x, real64 y, real64 z )
{
    Move( Vector3( x, y, z ) );
}

// set material
void Mesh::SetMaterial( const Handle<Material>& material )
{
    Geometry::SetMaterial( material );

    for ( auto& tri : _triangles )
    {
        tri->SetMaterial( _material );
    }
}

// set material
void Mesh::SetMaterial( const Material& material )
{
    Geometry::SetMaterial( material );

    for ( auto& tri : _triangles )
    {
        tri->SetMaterial( _material );
    }
}



// load mesh
bool Mesh::LoadFile( const String& fname, std::vector<Handle<Mesh>>& meshes )
{
    rex::Write( "  Reading '", fname, "'... " );
    
    // read the mesh file (generate normals in case we need to end up using those
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile( fname,
                                              aiProcess_JoinIdenticalVertices
                                            | aiProcess_ImproveCacheLocality
                                            | aiProcess_LimitBoneWeights
                                            | aiProcess_RemoveRedundantMaterials
                                            | aiProcess_SplitLargeMeshes
                                            | aiProcess_Triangulate
                                            | aiProcess_SortByPType
                                            | aiProcess_FindDegenerates
                                            | aiProcess_FindInvalidData
                                            | aiProcess_OptimizeMeshes
                                            | aiProcess_FindInstances
                                            | aiProcess_ValidateDataStructure
                                            | aiProcess_FixInfacingNormals
                                            );

    rex::WriteLine( "Done." );

    // ensure that the scene was read successfully
    if ( !scene->mFlags || !scene->mRootNode )
    {
        rex::WriteLine( "  Failed to load file as model." );
        return false;
    }

    // now process the scene
    meshes.clear();
    ProcessAssimpNode( scene->mRootNode, scene, meshes );

    return true;
}

// process an Assimp node
void Mesh::ProcessAssimpNode( aiNode* node, const aiScene* scene, std::vector<Handle<Mesh>>& meshes )
{
    // create a new mesh
    Handle<Mesh> mesh( new Mesh() );


    // sets the mesh's name
    mesh->_name = String( node->mName.data, node->mName.length );

    rex::WriteLine( "  Processing node '", mesh->_name, "'..." );
    rex::WriteLine( "    ", node->mNumMeshes, " meshes" );
    rex::WriteLine( "    ", node->mNumChildren, " children" );

    // now process the meshes in the node
    for ( uint32 i = 0; i < node->mNumMeshes; ++i )
    {
        aiMesh* aMesh = scene->mMeshes[ node->mMeshes[ i ] ];
        ProcessAssimpMesh( aMesh, scene, mesh );
    }

    // record the mesh
    meshes.push_back( mesh );


    // now process the rest of the nodes in the scene
    for ( uint32 i = 0; i < node->mNumChildren; ++i )
    {
        ProcessAssimpNode( node->mChildren[ i ], scene, meshes );
    }
}

// process an Assimp mesh
void Mesh::ProcessAssimpMesh( aiMesh* aMesh, const aiScene* scene, Handle<Mesh>& rMesh )
{
    rMesh->_triangles.clear();

    rex::WriteLine( "    Processing mesh '", aMesh->mName.data, "'..." );
    rex::WriteLine( "      ", aMesh->mNumFaces, " faces" );
    rex::WriteLine( "      ", aMesh->mNumVertices, " vertices" );

    // load all of the triangles from the given Assimp mesh into our mesh
    for ( uint32 i = 0; i < aMesh->mNumFaces; ++i )
    {
        Handle<Triangle> hTri( new Triangle() );

        // get the face information (we triangulated the faces so there are only 3 indices)
        aiFace face = aMesh->mFaces[ i ];
        if ( face.mNumIndices != 3 )
        {
            std::cout << "        Face #" << i << " is not a triangle (" << face.mNumIndices << " indices)" << std::endl;
            continue;
        }
        hTri->P1 = AIVectorToRexVector( aMesh->mVertices[ face.mIndices[ 0 ] ] );
        hTri->P2 = AIVectorToRexVector( aMesh->mVertices[ face.mIndices[ 1 ] ] );
        hTri->P3 = AIVectorToRexVector( aMesh->mVertices[ face.mIndices[ 2 ] ] );

        // now record the triangle
        rMesh->_triangles.push_back( hTri );
    }

    // build our mesh's octree data
    rex::Write( "      Building octree... " );
    rMesh->BuildOctree();
    rex::WriteLine( "Done." );
}

REX_NS_END