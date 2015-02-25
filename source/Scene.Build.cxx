#include <rex/Rex.hxx>
#include <rex/Debug.hxx>

REX_NS_BEGIN

#if 1

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
    _bgColor = Color( 0.05f );


    // setup view plane
    _viewPlane.Width        = hres;
    _viewPlane.Height       = vres;
    _viewPlane.PixelSize    = ps;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f / _viewPlane.InvGamma; // in case .Gamma is changed

    // setup the image
    _image.reset( new Image( hres, vres ) );


    // point lights
    //auto p1 = AddPointLight(   0.0,    0.0,  120.0 );
    //auto p2 = AddPointLight(   0.0, -100.0,    0.0 );
    //auto p3 = AddPointLight( 100.0,  100.0,    0.0 );
    //auto p4 = AddPointLight(   0.0,    0.0, -200.0 );
    //p2->SetColor( 1.00f, 0.00f, 0.50f );
    //p3->SetColor( 0.00f, 0.80f, 0.32f );
    //p4->SetColor( 0.10f, 0.40f, 0.80f );


    // directional lights
    auto d1 = AddDirectionalLight( 100.0, 100.0, 200.0 );
    d1->SetRadianceScale( 2.0 );



    // materials
    const real32 ka     = 0.25f;
    const real32 kd     = 0.75f;
    const real32 ks     = 0.20f;
    const real32 kpow   = 2.00f;
    PhongMaterial yellow      ( Color( 1.00f, 1.00f, 0.00f ), ka, kd, ks, kpow );
    PhongMaterial brown       ( Color( 0.71f, 0.40f, 0.16f ), ka, kd, ks, kpow );
    PhongMaterial dark_green  ( Color( 0.00f, 0.41f, 0.41f ), ka, kd, ks, kpow );
    PhongMaterial orange      ( Color( 1.00f, 0.75f, 0.00f ), ka, kd, ks, kpow );
    PhongMaterial green       ( Color( 0.00f, 0.60f, 0.30f ), ka, kd, ks, kpow );
    PhongMaterial light_green ( Color( 0.65f, 1.00f, 0.30f ), ka, kd, ks, kpow );
    PhongMaterial dark_yellow ( Color( 0.61f, 0.61f, 0.00f ), ka, kd, ks, kpow );
    PhongMaterial light_purple( Color( 0.65f, 0.30f, 1.00f ), ka, kd, ks, kpow );
    PhongMaterial dark_purple ( Color( 0.50f, 0.00f, 1.00f ), ka, kd, ks, kpow );

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

#endif
#if 0

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
    _bgColor = Color( 0.05f );


    // setup view plane
    _viewPlane.Width        = hres;
    _viewPlane.Height       = vres;
    _viewPlane.PixelSize    = ps;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f / _viewPlane.InvGamma; // in case .Gamma is changed

    // setup the image
    _image.reset( new Image( hres, vres ) );


    // point lights
    auto p1 = AddPointLight( -50.0, 50.0, 0.0 );
    auto p2 = AddPointLight(  50.0, 50.0, 0.0 );
    p1->SetColor( Color::Red  );
    p2->SetColor( Color::Blue );


    // materials
    const real32 ka   = 0.25f;
    const real32 kd   = 0.75f;
    const real32 ks   = 0.20f;
    const real32 kpow = 2.00f;
    MatteMaterial m   = MatteMaterial( Color::White, ka, kd );


    // add plane
    const Vector3 point  = Vector3( 0.0, 0.0, 0.0 );
    const Vector3 normal = Vector3( 0.0, 1.0, 0.0 );
    AddPlane( point, normal, m );
}

#endif

REX_NS_END