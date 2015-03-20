#include <rex/GL/GLWindowHints.hxx>

REX_NS_BEGIN

// create new hints
GLWindowHints::GLWindowHints()
{
    Resizable  = false;
    Visible    = false;
    Fullscreen = false;
    VSync      = true;
}

// destroy hints
GLWindowHints::~GLWindowHints()
{
    Resizable  = 0;
    Visible    = 0;
    Fullscreen = 0;
    VSync      = 0;
}

REX_NS_END