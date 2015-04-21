#include <rex/Rex.hxx>

REX_NS_BEGIN

// create a new scene
Scene::Scene( SceneRenderMode renderMode )
    : _lights    ( nullptr    ),
      _geometry  ( nullptr    ),
      _octree    ( nullptr    ),
      _texture   ( nullptr    ),
      _image     ( nullptr    ),
      _window    ( nullptr    ),
      _renderMode( renderMode )
{
}

// destroy this scene
Scene::~Scene()
{
    Dispose();
}

// saves this scene's image
void Scene::SaveImage( const char* fname ) const
{
    if ( _image )
    {
        _image->Save( fname );
    }
}

// set camera position
void Scene::SetCameraPosition( const Vector3& pos )
{
    _camera.SetPosition( pos );
}

// set camera position
void Scene::SetCameraPosition( real_t x, real_t y, real_t z )
{
    _camera.SetPosition( x, y, z );
}

REX_NS_END