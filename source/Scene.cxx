#include "Scene.hxx"

REX_NS_BEGIN

// new scene
Scene::Scene()
{
    _image.reset( new Image( 1, 1 ) );
    _tracer.reset( new RayTracer( this ) ); // "Null" ray tracer
}

// destroy scene
Scene::~Scene()
{
}

// get image
const Handle<Image>& Scene::GetImage() const
{
    return _image;
}

// get sphere
const Sphere& Scene::GetSphere() const
{
    return _sphere;
}

// build scene
void Scene::Build( int32 hres, int32 vres )
{
    // set the background color
    _bgColor = Color::Black;

    // setup view plane
    _viewPlane.Width = hres;
    _viewPlane.Height = vres;
    
    // setup the image
    _image.reset( new Image( hres, vres ) );

    // setup the sphere
    _sphere.Center = Vector3( 0.0 );
    _sphere.Radius = 85.0;
}

// render the scene
void Scene::Render()
{
    Color  color;
    Ray    ray;
    real64 zw = 100.0;
    real64 x, y;

    // set the ray's direction
    ray.Direction = Vector3( 0.0, 0.0, -1.0 );

    // go through each pixel and set the color
    for ( int32 py = 0; py < _image->GetHeight(); ++py )
    {
        for ( int32 px = 0; px < _image->GetWidth(); ++px )
        {
            // calculate the X and Y values for the ray
            x = _viewPlane.PixelSize * ( px - 0.5 * ( _viewPlane.Width  - 1.0 ) );
            y = _viewPlane.PixelSize * ( py - 0.5 * ( _viewPlane.Height - 1.0 ) );

            // set the ray's origin
            ray.Origin = Vector3( x, y, zw );

            // calculate the pixel color
            color = _tracer->Trace( ray );
            _image->SetPixelUnchecked( px, py, color );
        }
    }
}

REX_NS_END