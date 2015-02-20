#include "Rex.hxx"
#include "RegularSampler.hxx"
#include "PerspectiveCamera.hxx"

REX_NS_BEGIN

// new scene
Scene::Scene()
{
    _image.reset( new Image( 1, 1 ) );
    SetCameraType<PerspectiveCamera>();
    SetSamplerType<RegularSampler>();
    SetTracerType<Tracer>();
}

// destroy scene
Scene::~Scene()
{
}

// get background color.
const Color& Scene::GetBackgroundColor() const
{
    return _bgColor;
}

// get camera
const Handle<Camera>& Scene::GetCamera() const
{
    return _camera;
}

// get image
const Handle<Image>& Scene::GetImage() const
{
    return _image;
}

// get sampler
const Handle<Sampler>& Scene::GetSampler() const
{
    return _sampler;
}

// get tracer
const Handle<Tracer>& Scene::GetTracer() const
{
    return _tracer;
}

// get view plane
const ViewPlane& Scene::GetViewPlane() const
{
    return _viewPlane;
}

// hit all objects
void Scene::HitObjects( const Ray& ray, ShadePoint& sp ) const
{
    real64 t     = 0.0;
    real64 tmin  = Math::HUGE_VALUE;
    size_t count = _objects.size();

    // iterate through all objects
    for ( uint32 i = 0; i < count; ++i )
    {
        auto& obj = _objects[ i ];
        if ( obj->Hit( ray, t, sp ) && ( t < tmin ) )
        {
            sp.HasHit = true;
            sp.Color  = obj->GetColor();
            tmin      = t;
        }
    }
}

// add plane
void Scene::AddPlane( const Vector3& point, const Vector3& normal )
{
    AddPlane( point, normal, Color::Black );
}

// add plane w/ color
void Scene::AddPlane( const Vector3& point, const Vector3& normal, const Color& color )
{
    auto plane = Handle<Plane>( new Plane( point, normal, color ) );
    _objects.push_back( plane );
}

// adds a sphere
void Scene::AddSphere( const Vector3& center, real64 radius )
{
    AddSphere( center, radius, Color::Black );
}

// adds a sphere w/ color
void Scene::AddSphere( const Vector3& center, real64 radius, const Color& color )
{
    auto sphere = Handle<Sphere>( new Sphere( center, radius, color ) );
    _objects.push_back( sphere );
}

// build scene
void Scene::Build( int32 hres, int32 vres, real32 ps )
{
    // set the background color
    _bgColor = Color::Black;

    // setup view plane
    _viewPlane.Width        = hres;
    _viewPlane.Height       = vres;
    _viewPlane.PixelSize    = ps;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f;

    // setup the image
    _image.reset( new Image( hres, vres ) );


    // I did some MAJOR refactoring from the example...


    // colors
    Color yellow      ( 1.00f, 1.00f, 0.00f );
    Color brown       ( 0.71f, 0.40f, 0.16f );
    Color dark_green  ( 0.00f, 0.41f, 0.41f );
    Color orange      ( 1.00f, 0.75f, 0.00f );
    Color green       ( 0.00f, 0.60f, 0.30f );
    Color light_green ( 0.65f, 1.00f, 0.30f );
    Color dark_yellow ( 0.61f, 0.61f, 0.00f );
    Color light_purple( 0.65f, 0.30f, 1.00f );
    Color dark_purple ( 0.50f, 0.00f, 1.00f );

    // spheres
    AddSphere( Vector3(   5,   3,    0 ), 30, yellow        );
    AddSphere( Vector3(  45,  -7,  -60 ), 20, brown         );
    AddSphere( Vector3(  40,  43, -100 ), 17, dark_green    );
    AddSphere( Vector3( -20,  28,  -15 ), 20, orange        );
    AddSphere( Vector3( -25,  -7,  -35 ), 27, green         );
    AddSphere( Vector3(  20, -27,  -35 ), 25, light_green   );
    AddSphere( Vector3(  35,  18,  -35 ), 22, green         );
    AddSphere( Vector3( -57, -17,  -50 ), 15, brown         );
    AddSphere( Vector3( -47,  16,  -80 ), 23, light_green   );
    AddSphere( Vector3( -15, -32,  -60 ), 22, dark_green    );
    AddSphere( Vector3( -35, -37,  -80 ), 22, dark_yellow   );
    AddSphere( Vector3(  10,  43,  -80 ), 22, dark_yellow   );
    AddSphere( Vector3(  30,  -7,  -80 ), 10, dark_yellow   );
    AddSphere( Vector3( -40,  48, -110 ), 18, dark_green    );
    AddSphere( Vector3( -10,  53, -120 ), 18, brown         );
    AddSphere( Vector3( -55, -52, -100 ), 10, light_purple  );
    AddSphere( Vector3(   5, -52, -100 ), 15, brown         );
    AddSphere( Vector3( -20, -57, -120 ), 15, dark_purple   );
    AddSphere( Vector3(  55, -27, -100 ), 17, dark_green    );
    AddSphere( Vector3(  50, -47, -120 ), 15, brown         );
    AddSphere( Vector3(  70, -42, -150 ), 10, light_purple  );
    AddSphere( Vector3(   5,  73, -130 ), 12, light_purple  );
    AddSphere( Vector3(  66,  21, -130 ), 13, dark_purple   );
    AddSphere( Vector3(  72, -12, -140 ), 12, light_purple  );
    AddSphere( Vector3(  64,   5, -160 ), 11, green         );
    AddSphere( Vector3(  55,  38, -160 ), 12, light_purple  );
    AddSphere( Vector3( -73,  -2, -160 ), 12, light_purple  );
    AddSphere( Vector3(  30, -62, -140 ), 15, dark_purple   );
    AddSphere( Vector3(  25,  63, -140 ), 15, dark_purple   );
    AddSphere( Vector3( -60,  46, -140 ), 15, dark_purple   );
    AddSphere( Vector3( -30,  68, -130 ), 12, light_purple  );
    AddSphere( Vector3(  58,  56, -180 ), 11, green         );
    AddSphere( Vector3( -63, -39, -180 ), 11, green         );
    AddSphere( Vector3(  46,  68, -200 ), 10, light_purple  );
    AddSphere( Vector3(  -3, -72, -130 ), 12, light_purple  );
}

// get camera
Handle<Camera>& Scene::GetCamera()
{
    return _camera;
}

// get image
Handle<Image>& Scene::GetImage()
{
    return _image;
}

// get sampler
Handle<Sampler>& Scene::GetSampler()
{
    return _sampler;
}

// get tracer
Handle<Tracer>& Scene::GetTracer()
{
    return _tracer;
}

// render the scene
void Scene::Render()
{
    Scene& me = *this;
    _camera->Render( me );
}

REX_NS_END