#include <rex/Rex.hxx>
#include <GLFW/glfw3.h>

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
    //cudaSetDevice( 0 );
    //cudaGLSetGLDevice( 0 );
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
void Scene::SetCameraPosition( const vec3& pos )
{
    _camera.SetPosition( pos );
}

// set camera position
void Scene::SetCameraPosition( real32 x, real32 y, real32 z )
{
    _camera.SetPosition( x, y, z );
}

// update the scene camera
void Scene::UpdateCamera( real64 dt )
{
    static real64 oldMouseX = 0.0, oldMouseY = 0.0;
    static real64 newMouseX = 0.0, newMouseY = 0.0;

    // get local axes
    const vec3& xAxis = _camera.GetOrthoX();
    const vec3& yAxis = _camera.GetOrthoY();
    const vec3& zAxis = _camera.GetOrthoZ();

    // get helper variables
    vec3     translation;
    real32      moveSpeed = real32( 25.0 * dt );
    real32      rotSpeed  = real32(  5.0 * dt );
    GLFWwindow* window    = reinterpret_cast<GLFWwindow*>( _window->_handle );

    // check keys for movement
    bool hasMoved = false;
    if ( glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS )
    {
        translation += zAxis;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_W ) == GLFW_PRESS )
    {
        translation -= zAxis;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_D ) == GLFW_PRESS )
    {
        translation += xAxis;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS )
    {
        translation -= xAxis;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_SPACE ) == GLFW_PRESS )
    {
        translation += yAxis;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_LEFT_CONTROL ) == GLFW_PRESS )
    {
        translation -= yAxis;
        hasMoved = true;
    }

    // move
    if ( hasMoved )
    {
        translation = moveSpeed * glm::normalize( translation );
    }
    _camera.SetPosition( _camera.GetPosition() + translation );
}

REX_NS_END