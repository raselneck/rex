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
    static std::vector<const Geometry*> geom; // TODO : BAD! BAD!!


    // only get the objects that the ray hits
    _octree->QueryIntersections( ray, geom );


    // prepare to check objects
    real64 t     = 0.0;
    real64 tmin  = Math::HUGE_VALUE;
    size_t count = _objects.size();
    Vector3 normal;
    Vector3 localHitPoint;


    


    // start the timer
    Timer timer; timer.Start();

    // iterate through the hit objects
    for ( auto& obj : geom )
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

    // stop timer and print time, if applicable
    timer.Stop(); real64 time = timer.GetElapsed();
    if ( time > 0.0 )
    {
        rex::WriteLine( "checking all objects took ", time, " seconds" );
    }





    // restore hit point data from closest object
    if ( sp.HasHit )
    {
        sp.T = tmin;
        sp.Normal = normal;
        sp.LocalHitPoint = localHitPoint;
    }
}

// shadow-hit all objects
bool Scene::ShadowHitObjects( const Ray& ray ) const
{
    real64 t = 0.0;

    for ( auto& obj : _objects )
    {
        if ( obj->ShadowHit( ray, t ) )
        {
            return true;
        }
    }

    return false;
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

// build octree bounds
void Scene::BuildOctree()
{
    // TODO : This does not work with planes

    // get the absolute min and max of the scene
    Vector3 min, max;
    for ( auto& obj : _objects )
    {
        BoundingBox bb = obj->GetBounds();
        min = Vector3::Min( min, bb.GetMin() );
        max = Vector3::Max( max, bb.GetMax() );
    }

    // create our octree
    _octree.reset( new Octree( min, max ) );

    // add all objects' bounds to the octree
    for ( auto& obj : _objects )
    {
        _octree->Add( obj.get() );
    }
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