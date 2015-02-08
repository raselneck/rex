#include "Scene.hxx"
#include "Plane.hxx"
#include "Sphere.hxx"

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

// get background color.
const Color& Scene::GetBackgroundColor() const
{
    return _bgColor;
}

// get image
const Handle<Image>& Scene::GetImage() const
{
    return _image;
}

// hit all objects
ShadePoint Scene::HitObjects( const Ray& ray ) const
{
    ShadePoint sp( const_cast<Scene*>( this ) );
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
            sp.Color  = obj->Color;
            tmin      = t;
        }
    }

    return sp;
}

// adds a plane
void Scene::AddPlane( const Vector3& point, const Vector3& normal, const Color& color )
{
    auto plane = Handle<Plane>( new Plane( point, normal ) );
    plane->Color = color;
    _objects.push_back( plane );
}

// adds a sphere
void Scene::AddSphere( const Vector3& center, real64 radius, const Color& color )
{
    auto sphere = Handle<Sphere>( new Sphere( center, radius ) );
    sphere->Color = color;
    _objects.push_back( sphere );
}

// build scene
void Scene::Build( int32 hres, int32 vres, real32 ps )
{
    // set the background color
    _bgColor = Color::Black;

    // setup view plane
    _viewPlane.Width = hres;
    _viewPlane.Height = vres;
    _viewPlane.PixelSize = ps;
    _viewPlane.Gamma = 1.0f;
    _viewPlane.InvGamma = 1.0f;
    
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

#define add_object(s) AddSphere(s.Center, s.Radius, s.Color)

    // spheres
    Sphere sphere_ptr1( Vector3( 5, 3, 0 ), 30 );
    sphere_ptr1.Color = ( yellow );
    add_object( sphere_ptr1 );

    Sphere sphere_ptr2( Vector3( 45, -7, -60 ), 20 );
    sphere_ptr2.Color = ( brown );
    add_object( sphere_ptr2 );

    Sphere sphere_ptr3( Vector3( 40, 43, -100 ), 17 );
    sphere_ptr3.Color = ( dark_green );
    add_object( sphere_ptr3 );

    Sphere sphere_ptr4( Vector3( -20, 28, -15 ), 20 );
    sphere_ptr4.Color = ( orange );
    add_object( sphere_ptr4 );

    Sphere sphere_ptr5( Vector3( -25, -7, -35 ), 27 );
    sphere_ptr5.Color = ( green );
    add_object( sphere_ptr5 );

    Sphere sphere_ptr6( Vector3( 20, -27, -35 ), 25 );
    sphere_ptr6.Color = ( light_green );
    add_object( sphere_ptr6 );

    Sphere sphere_ptr7( Vector3( 35, 18, -35 ), 22 );
    sphere_ptr7.Color = ( green );
    add_object( sphere_ptr7 );

    Sphere sphere_ptr8( Vector3( -57, -17, -50 ), 15 );
    sphere_ptr8.Color = ( brown );
    add_object( sphere_ptr8 );

    Sphere sphere_ptr9( Vector3( -47, 16, -80 ), 23 );
    sphere_ptr9.Color = ( light_green );
    add_object( sphere_ptr9 );

    Sphere sphere_ptr10( Vector3( -15, -32, -60 ), 22 );
    sphere_ptr10.Color = ( dark_green );
    add_object( sphere_ptr10 );

    Sphere sphere_ptr11( Vector3( -35, -37, -80 ), 22 );
    sphere_ptr11.Color = ( dark_yellow );
    add_object( sphere_ptr11 );

    Sphere sphere_ptr12( Vector3( 10, 43, -80 ), 22 );
    sphere_ptr12.Color = ( dark_yellow );
    add_object( sphere_ptr12 );

    Sphere sphere_ptr13( Vector3( 30, -7, -80 ), 10 );
    sphere_ptr13.Color = ( dark_yellow );
    add_object( sphere_ptr13 );

    Sphere sphere_ptr14( Vector3( -40, 48, -110 ), 18 );
    sphere_ptr14.Color = ( dark_green );
    add_object( sphere_ptr14 );

    Sphere sphere_ptr15( Vector3( -10, 53, -120 ), 18 );
    sphere_ptr15.Color = ( brown );
    add_object( sphere_ptr15 );

    Sphere sphere_ptr16( Vector3( -55, -52, -100 ), 10 );
    sphere_ptr16.Color = ( light_purple );
    add_object( sphere_ptr16 );

    Sphere sphere_ptr17( Vector3( 5, -52, -100 ), 15 );
    sphere_ptr17.Color = ( brown );
    add_object( sphere_ptr17 );

    Sphere sphere_ptr18( Vector3( -20, -57, -120 ), 15 );
    sphere_ptr18.Color = ( dark_purple );
    add_object( sphere_ptr18 );

    Sphere sphere_ptr19( Vector3( 55, -27, -100 ), 17 );
    sphere_ptr19.Color = ( dark_green );
    add_object( sphere_ptr19 );

    Sphere sphere_ptr20( Vector3( 50, -47, -120 ), 15 );
    sphere_ptr20.Color = ( brown );
    add_object( sphere_ptr20 );

    Sphere sphere_ptr21( Vector3( 70, -42, -150 ), 10 );
    sphere_ptr21.Color = ( light_purple );
    add_object( sphere_ptr21 );

    Sphere sphere_ptr22( Vector3( 5, 73, -130 ), 12 );
    sphere_ptr22.Color = ( light_purple );
    add_object( sphere_ptr22 );

    Sphere sphere_ptr23( Vector3( 66, 21, -130 ), 13 );
    sphere_ptr23.Color = ( dark_purple );
    add_object( sphere_ptr23 );

    Sphere sphere_ptr24( Vector3( 72, -12, -140 ), 12 );
    sphere_ptr24.Color = ( light_purple );
    add_object( sphere_ptr24 );

    Sphere sphere_ptr25( Vector3( 64, 5, -160 ), 11 );
    sphere_ptr25.Color = ( green ); 
    add_object( sphere_ptr25 );

    Sphere sphere_ptr26( Vector3( 55, 38, -160 ), 12 );
    sphere_ptr26.Color = ( light_purple );
    add_object( sphere_ptr26 );

    Sphere sphere_ptr27( Vector3( -73, -2, -160 ), 12 );
    sphere_ptr27.Color = ( light_purple );
    add_object( sphere_ptr27 );

    Sphere sphere_ptr28( Vector3( 30, -62, -140 ), 15 );
    sphere_ptr28.Color = ( dark_purple );
    add_object( sphere_ptr28 );

    Sphere sphere_ptr29( Vector3( 25, 63, -140 ), 15 );
    sphere_ptr29.Color = ( dark_purple );
    add_object( sphere_ptr29 );

    Sphere sphere_ptr30( Vector3( -60, 46, -140 ), 15 );
    sphere_ptr30.Color = ( dark_purple );
    add_object( sphere_ptr30 );

    Sphere sphere_ptr31( Vector3( -30, 68, -130 ), 12 );
    sphere_ptr31.Color = ( light_purple );
    add_object( sphere_ptr31 );

    Sphere sphere_ptr32( Vector3( 58, 56, -180 ), 11 );
    sphere_ptr32.Color = ( green );
    add_object( sphere_ptr32 );

    Sphere sphere_ptr33( Vector3( -63, -39, -180 ), 11 );
    sphere_ptr33.Color = ( green );
    add_object( sphere_ptr33 );

    Sphere sphere_ptr34( Vector3( 46, 68, -200 ), 10 );
    sphere_ptr34.Color = ( light_purple );
    add_object( sphere_ptr34 );

    Sphere sphere_ptr35( Vector3( -3, -72, -130 ), 12 );
    sphere_ptr35.Color = ( light_purple );
    add_object( sphere_ptr35 );
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

    int32 num = 0;

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
            if ( _viewPlane.Gamma != 1.0f )
            {
                color = Color::Pow( color, _viewPlane.InvGamma );
            }

            // now set the pixel in the image
            _image->SetPixelUnchecked( px, py, color );


#if __RELEASE__
            // save a frame for each colored pixel
            if ( color != _bgColor )
            {
                std::string fname = "output/img" + std::to_string( num ) + ".png";
                _image->Save( fname.c_str() );
                ++num;
            }
#endif
        }
    }
}

REX_NS_END