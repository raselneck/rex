#include <rex/Utility/Image.hxx>
#include <rex/Utility/Math.hxx>

// include stb image write header
#pragma warning( push )
#pragma warning( disable : 4996 )
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma warning( pop )

REX_NS_BEGIN

// new image
Image::Image( uint16 width, uint16 height )
    : _width( width ),
      _height( height )
{
    _pixels.resize( _width*_height );
}

// destroy image
Image::~Image()
{
    const_cast<uint16&>( _width ) = 0;
    const_cast<uint16&>( _height ) = 0;
}

// get pixels
const Color* Image::GetPixels() const
{
    if ( _pixels.size() == 0 )
    {
        return nullptr;
    }
    return &( _pixels[ 0 ] );
}

// get pixel, checked
const Color* Image::GetPixel( uint32 x, uint32 y ) const
{
    if ( x < _width && y < _height )
    {
        return GetPixelUnchecked( x, y );
    }
    return nullptr;
}

// get pixel, unchecked
const Color* Image::GetPixelUnchecked( uint32 x, uint32 y ) const
{
    return &( _pixels[ x + y * _width ] );
}

// get width
uint16 Image::GetWidth() const
{
    return _width;
}

// get height
uint16 Image::GetHeight() const
{
    return _height;
}

// save image
bool Image::Save( const char* fname ) const
{
    // convert our floating-point colors into 8-bit colors
    std::vector<uint8> converted;
    converted.resize( _width*_height * 3 );
    size_t convInd = 0;
    for ( auto iter = _pixels.begin(); iter != _pixels.end(); ++iter, convInd += 3 )
    {
        converted[ convInd + 0 ] = static_cast<uint8>( Math::Clamp( iter->R, 0.0f, 1.0f ) * 255 );
        converted[ convInd + 1 ] = static_cast<uint8>( Math::Clamp( iter->G, 0.0f, 1.0f ) * 255 );
        converted[ convInd + 2 ] = static_cast<uint8>( Math::Clamp( iter->B, 0.0f, 1.0f ) * 255 );
    }

    // now write out the image as a PNG
    return 0 == stbi_write_png( fname, _width, _height, 3, &( converted[ 0 ] ), _width * 3 );
}

// get pixels
Color* Image::GetPixels()
{
    if ( _pixels.size() == 0 )
    {
        return nullptr;
    }
    return &( _pixels[ 0 ] );
}

// get pixel, checked
Color* Image::GetPixel( uint32 x, uint32 y )
{
    if ( x < _width && y < _height )
    {
        return GetPixelUnchecked( x, y );
    }
    return nullptr;
}

// get pixel, unchecked
Color* Image::GetPixelUnchecked( uint32 x, uint32 y )
{
    return &( _pixels[ x + y * _width ] );
}

// set pixel, checked
void Image::SetPixel( uint32 x, uint32 y, const Color& color )
{
    if ( x < _width && y < _height )
    {
        SetPixelUnchecked( x, y, color );
    }
}

// set pixel, checked
void Image::SetPixel( uint32 x, uint32 y, real32 r, real32 g, real32 b )
{
    if ( x < _width && y < _height )
    {
        Color color( r, g, b );
        SetPixelUnchecked( x, y, color );
    }
}

// set pixel, unchecked
void Image::SetPixelUnchecked( uint32 x, uint32 y, const Color& color )
{
    _pixels[ x + y * _width ] = color;
}

// set pixel, unchecked
void Image::SetPixelUnchecked( uint32 x, uint32 y, real32 r, real32 g, real32 b )
{
    Color color( r, g, b );
    SetPixelUnchecked( x, y, color );
}

REX_NS_END