#include <rex/Rex.hxx>
#include <rex/Debug.hxx>

REX_NS_BEGIN

// new scene
Scene::Scene()
{
    _image.reset( new Image( 1, 1 ) );
    _ambientLight.reset( new AmbientLight() );

    // no default values for these
    //SetCameraType<PerspectiveCamera>();
    //SetSamplerType<RegularSampler>();
    //SetTracerType<Tracer>();
}

// destroy scene
Scene::~Scene()
{
}

// get ambient color
const Color& Scene::GetAmbientColor() const
{
    return _ambientLight->GetColor();
}

// get ambient radiance
Color Scene::GetAmbientRadiance() const
{
    ShadePoint sp( nullptr );
    return _ambientLight->GetRadiance( sp );
}

// get ambient radiance scale
real32 Scene::GetAmbientRadianceScale() const
{
    return _ambientLight->GetRadianceScale();
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

// get lights
const std::vector<Handle<Light>>& Scene::GetLights() const
{
    return _lights;
}

// get light count
uint32 Scene::GetLightCount() const
{
    return static_cast<uint32>( _lights.size() );
}

// get objects
const std::vector<Handle<Geometry>>& Scene::GetObjects() const
{
    return _objects;
}

// get object count
uint32 Scene::GetObjectCount() const
{
    return static_cast<uint32>( _objects.size() );
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
    Vector3 normal;
    Vector3 localHitPoint;

    // iterate through all objects
    for ( uint32 i = 0; i < count; ++i )
    {
        auto& obj = _objects[ i ];
        if ( obj->Hit( ray, t, sp ) && ( t < tmin ) )
        {
            sp.HasHit     = true;
            sp.Material   = const_cast<Material*>( obj->GetMaterial() );
            sp.HitPoint   = ray.Origin + t * ray.Direction;

            tmin          = t;
            normal        = sp.Normal;
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
}

// add directional light
Handle<DirectionalLight> Scene::AddDirectionalLight( const Vector3& direction )
{
    Handle<DirectionalLight> light( new DirectionalLight( direction ) );
    _lights.push_back( light );
    return light;
}

// add directional light
Handle<DirectionalLight> Scene::AddDirectionalLight( real64 x, real64 y, real64 z )
{
    return AddDirectionalLight( Vector3( x, y, z ) );
}

// add plane
Handle<Plane> Scene::AddPlane( const Plane& plane )
{
    return AddPlane( plane.GetPoint(), plane.GetNormal() );
}

// add plane
Handle<Plane> Scene::AddPlane( const Vector3& point, const Vector3& normal )
{
    Handle<Plane> plane( new Plane( point, normal ) );
    _objects.push_back( plane );
    return plane;
}

// add point light
Handle<PointLight> Scene::AddPointLight( const Vector3& position )
{
    Handle<PointLight> light( new PointLight( position ) );
    _lights.push_back( light );
    return light;
}

// add point light
Handle<PointLight> Scene::AddPointLight( real64 x, real64 y, real64 z )
{
    return AddPointLight( Vector3( x, y, z ) );
}

// adds a sphere
Handle<Sphere> Scene::AddSphere( const Sphere& sphere )
{
    return AddSphere( sphere.GetCenter(), sphere.GetRadius() );
}

// adds a sphere
Handle<Sphere> Scene::AddSphere( const Vector3& center, real64 radius )
{
    Handle<Sphere> sphere( new Sphere( center, radius ) );
    _objects.push_back( sphere );
    return sphere;
}

// build scene
void Scene::Build( int32 hres, int32 vres, real32 ps )
{
    // ensure we have a sampler
    if ( !_sampler )
    {
        rex::WriteLine( "The sampler must be set before building." );
        return;
    }


    // set the background color
    _bgColor = Color::Black;

    // setup view plane
    _viewPlane.Width        = hres;
    _viewPlane.Height       = vres;
    _viewPlane.PixelSize    = ps;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f / _viewPlane.InvGamma;

    // setup the image
    _image.reset( new Image( hres, vres ) );


    // point lights
    auto p1 = AddPointLight(   0.0,    0.0,  120.0 );
    auto p2 = AddPointLight(   0.0, -100.0,    0.0 );
    auto p3 = AddPointLight( 100.0,  100.0,    0.0 );
    auto p4 = AddPointLight(   0.0,    0.0, -200.0 );
    p2->SetColor( 1.00f, 0.00f, 0.50f );
    p3->SetColor( 0.00f, 0.80f, 0.32f );
    p4->SetColor( 0.10f, 0.40f, 0.80f );


    // directional lights
    auto d1 = AddDirectionalLight( 100.0, 100.0, 200.0 );
    d1->SetRadianceScale( 2.0 );



    // materials
    const real32 ka = 0.25f;
    const real32 kd = 0.75f;
    MatteMaterial yellow      ( Color( 1.00f, 1.00f, 0.00f ), ka, kd );
    MatteMaterial brown       ( Color( 0.71f, 0.40f, 0.16f ), ka, kd );
    MatteMaterial dark_green  ( Color( 0.00f, 0.41f, 0.41f ), ka, kd );
    MatteMaterial orange      ( Color( 1.00f, 0.75f, 0.00f ), ka, kd );
    MatteMaterial green       ( Color( 0.00f, 0.60f, 0.30f ), ka, kd );
    MatteMaterial light_green ( Color( 0.65f, 1.00f, 0.30f ), ka, kd );
    MatteMaterial dark_yellow ( Color( 0.61f, 0.61f, 0.00f ), ka, kd );
    MatteMaterial light_purple( Color( 0.65f, 0.30f, 1.00f ), ka, kd );
    MatteMaterial dark_purple ( Color( 0.50f, 0.00f, 1.00f ), ka, kd );

    // set material samplers
    yellow      .SetSampler( _sampler );
    brown       .SetSampler( _sampler );
    dark_green  .SetSampler( _sampler );
    orange      .SetSampler( _sampler );
    green       .SetSampler( _sampler );
    light_green .SetSampler( _sampler );
    dark_yellow .SetSampler( _sampler );
    light_purple.SetSampler( _sampler );
    dark_purple .SetSampler( _sampler );


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
    // ensure we can actually render
    bool canRender = true; // so we can have multiple errors
    if ( !_ambientLight )
    {
        rex::WriteLine( "Cannot render scene without an ambient light" );
        canRender = false;
    }
    if ( !_camera )
    {
        rex::WriteLine( "Cannot render scene without a camera" );
        canRender = false;
    }
    if ( !_image )
    {
        rex::WriteLine( "Cannot render scene without an image" );
        canRender = false;
    }
    if ( !_sampler )
    {
        rex::WriteLine( "Cannot render scene without a sampler" );
        canRender = false;
    }
    if ( !_tracer )
    {
        rex::WriteLine( "Cannot render scene without a ray tracer" );
        canRender = false;
    }

    // now render if we actually can
    if ( canRender )
    {
        Scene& me = *this;
        _camera->Render( me );
    }
}

// set ambient color
void Scene::SetAmbientColor( const Color& color )
{
    _ambientLight->SetColor( color );
}

// set ambient color
void Scene::SetAmbientColor( real32 r, real32 g, real32 b )
{
    _ambientLight->SetColor( r, g, b );
}

// set ambient radiance scale
void Scene::SetAmbientRadianceScale( real32 ls )
{
    _ambientLight->SetRadianceScale( ls );
}

REX_NS_END