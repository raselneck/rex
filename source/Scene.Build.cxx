#include <rex/Rex.hxx>
#include <rex/Debug.hxx>

REX_NS_BEGIN

// sphere "grid"
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
    _bgColor = Color( 0.0725f );


    // setup view plane
    _viewPlane.Width = hres;
    _viewPlane.Height = vres;
    _viewPlane.PixelSize = ps;
    _viewPlane.Gamma = 1.0f;
    _viewPlane.InvGamma = 1.0f / _viewPlane.InvGamma; // in case .Gamma is changed


    // setup the image
    _image.reset( new Image( hres, vres ) );


    // directional lights
    auto d1 = AddDirectionalLight( 1.0, 1.0, 2.0 );
    d1->SetRadianceScale( 2.0 );


    // materials
    const real32 ka      = 0.25f;
    const real32 kd      = 0.75f;
    const real32 ks      = 0.30f;
    const real32 kpow    = 2.00f;
    const PhongMaterial  white      ( Color( 1.00f, 1.00f, 1.00f ), ka, kd, ks, kpow );
    const PhongMaterial  yellow     ( Color( 1.00f, 1.00f, 0.00f ), ka, kd, ks, kpow );
    const PhongMaterial  brown      ( Color( 0.71f, 0.40f, 0.16f ), ka, kd, ks, kpow );
    const PhongMaterial  darkGreen  ( Color( 0.00f, 0.41f, 0.41f ), ka, kd, ks, kpow );
    const PhongMaterial  orange     ( Color( 1.00f, 0.75f, 0.00f ), ka, kd, ks, kpow );
    const PhongMaterial  green      ( Color( 0.00f, 0.60f, 0.30f ), ka, kd, ks, kpow );
    const PhongMaterial  lightGreen ( Color( 0.65f, 1.00f, 0.30f ), ka, kd, ks, kpow );
    const PhongMaterial  darkYellow ( Color( 0.61f, 0.61f, 0.00f ), ka, kd, ks, kpow );
    const PhongMaterial  lightPurple( Color( 0.65f, 0.30f, 1.00f ), ka, kd, ks, kpow );
    const PhongMaterial  darkPurple ( Color( 0.50f, 0.00f, 1.00f ), ka, kd, ks, kpow );
    const PhongMaterial* materials[] =
    {
        &yellow,     &brown,       &darkGreen,
        &orange,     &green,       &lightGreen,
        &darkYellow, &lightPurple, &darkPurple,
        &white
    };
    const int32 materialCount = sizeof( materials ) / sizeof( materials[ 0 ] );
    rex::WriteLine( "  Using ", materialCount, " materials." );


    // add some randomized spheres into a grid
    const int32  dx = 75;
    const int32  dy = 75;
    const int32  dz = 75;
    const real32 maxR = Math::Min( Math::Min( dx, dy ), dz ) * 0.4f;
    uint32 sphereCount = 0;
    for ( int32 x = -250; x <= 250; x += dx )
    {
        for ( int32 y = -150; y <= 150; y += dy )
        {
            for ( int32 z = -250; z <= 100; z += dz, ++sphereCount )
            {
                // get random radius
                real32 radius = Random::RandReal32( 5.0f, maxR );

                // get random material
                const int32 matIndex = Random::RandInt32( 0, materialCount - 1 );
                const PhongMaterial& mat = *materials[ matIndex ];

                // add the sphere
                AddSphere( Vector3( x, y, z ), radius, mat );
            }
        }
    }
    rex::WriteLine( "  Generated ", sphereCount, " spheres." );
}

#endif

// randomly placed spheres
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
    _bgColor = Color( 0.0725f );


    // setup view plane
    _viewPlane.Width        = hres;
    _viewPlane.Height       = vres;
    _viewPlane.PixelSize    =   ps;
    _viewPlane.Gamma        = 1.0f;
    _viewPlane.InvGamma     = 1.0f / _viewPlane.InvGamma; // in case .Gamma is changed


    // setup the image
    _image.reset( new Image( hres, vres ) );


    // directional lights
    auto d1 = AddDirectionalLight( 1.0, 1.0, 2.0 );
    d1->SetRadianceScale( 2.0 );


    // materials
    const real32 ka     = 0.25f;
    const real32 kd     = 0.75f;
    const real32 ks     = 0.30f;
    const real32 kpow   = 2.00f;
    const PhongMaterial  white      ( Color( 1.00f, 1.00f, 1.00f ), ka, kd, ks, kpow );
    const PhongMaterial  yellow     ( Color( 1.00f, 1.00f, 0.00f ), ka, kd, ks, kpow );
    const PhongMaterial  brown      ( Color( 0.71f, 0.40f, 0.16f ), ka, kd, ks, kpow );
    const PhongMaterial  darkGreen  ( Color( 0.00f, 0.41f, 0.41f ), ka, kd, ks, kpow );
    const PhongMaterial  orange     ( Color( 1.00f, 0.75f, 0.00f ), ka, kd, ks, kpow );
    const PhongMaterial  green      ( Color( 0.00f, 0.60f, 0.30f ), ka, kd, ks, kpow );
    const PhongMaterial  lightGreen ( Color( 0.65f, 1.00f, 0.30f ), ka, kd, ks, kpow );
    const PhongMaterial  darkYellow ( Color( 0.61f, 0.61f, 0.00f ), ka, kd, ks, kpow );
    const PhongMaterial  lightPurple( Color( 0.65f, 0.30f, 1.00f ), ka, kd, ks, kpow );
    const PhongMaterial  darkPurple ( Color( 0.50f, 0.00f, 1.00f ), ka, kd, ks, kpow );
    const PhongMaterial* materials[] =
    {
        &yellow,     &brown,       &darkGreen,
        &orange,     &green,       &lightGreen,
        &darkYellow, &lightPurple, &darkPurple,
        &white
    };
    const int32 materialCount = sizeof( materials ) / sizeof( materials[ 0 ] );
    rex::WriteLine( "  Using ", materialCount, " materials." );
    

    // add some random spheres
    const uint32 sphereCount = static_cast<uint32>( Random::RandInt32( 100, 175 ) );
    rex::WriteLine( "  Generating ", sphereCount, " spheres." );

    Vector3 pos;
    for ( uint32 i = 0; i < sphereCount; ++i )
    {
        // randomize position
        pos.X = Random::RandReal32( -250.0f,  250.0f );
        pos.Y = Random::RandReal32( -150.0f,  150.0f );
        pos.Z = Random::RandReal32(  100.0f, -250.0f );
        
        // get random radius
        real32 radius = Random::RandReal32( 5.0f, 25.0f );

        // get random material
        const int32 matIndex = Random::RandInt32( 0, materialCount - 1 );
        const PhongMaterial& mat = *materials[ matIndex ];

        // add sphere
        AddSphere( pos, radius, mat );
    }
}

#endif

// sphere scene from the book
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

// blank slate w/ plane on XY plane
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