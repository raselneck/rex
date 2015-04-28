#include <rex/Rex.hxx>
#include <GLFW/glfw3.h>

using namespace glm;

REX_NS_BEGIN

// create a new scene
Scene::Scene( SceneRenderMode renderMode )
    : _lights    ( nullptr    )
    , _geometry  ( nullptr    )
    , _octree    ( nullptr    )
    , _texture   ( nullptr    )
    , _image     ( nullptr    )
    , _window    ( nullptr    )
    , _renderMode( renderMode )
{
}

// destroy this scene
Scene::~Scene()
{
    Dispose();
}

// handle a GLFW window key press
void Scene::OnKeyPress( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    // get the scene controlling the window
    Scene* scene = reinterpret_cast<Scene*>( glfwGetWindowUserPointer( window ) );
    
    // check if escape was pressed
    if ( action == GLFW_PRESS && key == GLFW_KEY_ESCAPE )
    {
        scene->_window->Close();
    }
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
    _camera.MoveTo( pos );
}

// set camera position
void Scene::SetCameraPosition( real32 x, real32 y, real32 z )
{
    _camera.MoveTo( x, y, z );
}

// update the scene camera
void Scene::UpdateCamera( real64 dt )
{
    static real64 oldMouseX = 0.0, oldMouseY = 0.0;
    static real64 newMouseX = 0.0, newMouseY = 0.0;
    static bool   isFirstUpdate = true;

    // get the GLFW window
    GLFWwindow* window = reinterpret_cast<GLFWwindow*>( _window->_handle );

    // check if this is the first update (and the mouse coordinates aren't both 0)
    glfwGetCursorPos( window, &newMouseX, &newMouseY );
    if ( isFirstUpdate && newMouseX * newMouseY != 0 )
    {
        oldMouseX = newMouseX;
        oldMouseY = newMouseY;
        isFirstUpdate = false;
    }



    // get helper variables
    vec3   translation;
    real32 moveSpeed = real32( 15.00 * dt );
    real32 rotSpeed  = real32(  0.03 * dt );

    // if left shift is down, speed up
    if ( glfwGetKey( window, GLFW_KEY_LEFT_SHIFT ) == GLFW_PRESS )
    {
        moveSpeed *= 2.5f;
    }

    // check keys for movement
    bool hasMoved = false;
    if ( glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS )
    {
        translation.z += moveSpeed;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_W ) == GLFW_PRESS )
    {
        translation.z -= moveSpeed;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_D ) == GLFW_PRESS )
    {
        translation.x += moveSpeed;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS )
    {
        translation.x -= moveSpeed;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_SPACE ) == GLFW_PRESS )
    {
        translation.y += moveSpeed;
        hasMoved = true;
    }
    if ( glfwGetKey( window, GLFW_KEY_LEFT_CONTROL ) == GLFW_PRESS )
    {
        translation.y -= moveSpeed;
        hasMoved = true;
    }


    // rotate based on the mouse
    glfwGetCursorPos( window, &newMouseX, &newMouseY );
    vec2 rotate = vec2( real32( newMouseY - oldMouseY ) * rotSpeed,
                        real32( newMouseX - oldMouseX ) * rotSpeed );
    _camera.Rotate( rotate );
    oldMouseX = newMouseX;
    oldMouseY = newMouseY;


    // move if we have moved
    if ( hasMoved )
    {
        translation = moveSpeed * normalize( translation );
        if ( !isnan( translation.x ) && !isnan( translation.y ) && !isnan( translation.z ) )
        {
            _camera.Move( translation );
        }
    }
}

REX_NS_END