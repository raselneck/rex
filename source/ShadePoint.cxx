#include "ShadePoint.hxx"

REX_NS_BEGIN

// new shade point
ShadePoint::ShadePoint( Scene* scene )
    : Color( Color::Black ),
      ScenePtr( scene ),
      HasHit( false )
{
}

// destroy shade point
ShadePoint::~ShadePoint()
{
    HasHit  = 0;
    const_cast<Scene*>( ScenePtr ) = 0;
}

// reset shade point
void ShadePoint::Reset()
{
    HitPoint.X = HitPoint.Y = HitPoint.Z = 0.0;
    Normal.X   = Normal.Y   = Normal.Z   = 0.0;
    Color.R    = Color.G    = Color.B    = 0.0;
    HasHit     = false;
}

REX_NS_END