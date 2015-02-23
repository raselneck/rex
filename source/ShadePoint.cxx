#include <rex/Scene/ShadePoint.hxx>

REX_NS_BEGIN

// new shade point
ShadePoint::ShadePoint( rex::Scene* scene )
    : Scene( scene ),
      Material( nullptr ),
      T( 0.0 ),
      RecursDepth( 0 ),
      HasHit( false )
{
}

// destroy shade point
ShadePoint::~ShadePoint()
{
    T           = 0.0;
    HasHit      = 0;
    RecursDepth = 0;
    Material    = nullptr;

    rex::Scene** pScene = const_cast<rex::Scene**>( &Scene );
    *pScene = 0;
}

// reset shade point
void ShadePoint::Reset()
{
    HitPoint.X = HitPoint.Y = HitPoint.Z = 0.0;
    Normal.X   = Normal.Y   = Normal.Z   = 0.0;
    Material   = nullptr;
    HasHit     = false;
}

REX_NS_END