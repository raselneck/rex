#include <rex/Geometry/Octree.hxx>

REX_NS_BEGIN

// the maximum number of children an octree can have before it subdivides
static const uint32 MaxOctreeChildCount = 16;

// viciously deletes octree children
static void DeleteOctreeChildren( Octree* children )
{
    delete[] children;
}




REX_NS_END