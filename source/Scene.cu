#include <rex/Rex.hxx>

REX_NS_BEGIN

// create a new scene
Scene::Scene()
{
}

// destroy this scene
Scene::~Scene()
{
}

// build the scene
bool Scene::Build( uint16 width, uint16 height )
{
    // make sure the image isn't too large
    if ( width > 1024 || height > 1024 )
    {
        Logger::Log( "Image is too large. Max dimensions are 1024x1024, given ", width, "x", height, "." );
        return false;
    }

    // create the image
    _image.reset( new Image( width, height ) );

    // set the background color
    _backgroundColor = Color( 0.0725f );

    // setup the view plane
    _viewPlane.Width        = width;
    _viewPlane.Height       = height;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f / _viewPlane.Gamma;
    _viewPlane.SampleCount  = 4;


    // add some lights
    auto d1 = _lights.AddDirectionalLight( 1.0, 1.0, 2.0 );
    d1->SetColor( Color::Red() );
    d1->SetRadianceScale( 1.5f );

    // prepare a material
    const real32 ka     = 0.25f;
    const real32 kd     = 0.75f;
    const real32 ks     = 0.30f;
    const real32 kpow   = 2.00f;
    const PhongMaterial white( Color( 1.00f, 1.00f, 1.00f ), ka, kd, ks, kpow );

    // add a sphere
    auto s1 = _geometry.AddSphere( Vector3(), 10.0f );
    s1->SetMaterial( white );


    // calculate the min and max of the bounds
    Vector3 min, max;
    for ( auto& shape : _geometry.GetGeometry() )
    {
        min = Vector3::Min( min, shape->GetBounds().GetMin() );
        max = Vector3::Max( max, shape->GetBounds().GetMax() );
    }

    // create the octree
    _octree.reset( new Octree( min, max ) );


    // configure the camera
    _camera.SetPosition( 0.0, 0.0, 40.0  );
    _camera.SetTarget  ( 0.0, 0.0, 0.0   );
    _camera.SetUp      ( 0.0, 1.0, 0.0   );
    _camera.SetViewPlaneDistance( 2000.0 );
    _camera.CalculateOrthonormalVectors();

    return true;
}

REX_NS_END