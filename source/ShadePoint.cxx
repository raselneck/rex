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

REX_NS_END