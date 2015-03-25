#include <rex/Utility/Image.hxx>
#include <rex/Math/Math.hxx>
#include <thrust/copy.h>

// include STB image write header
#pragma warning( push )
#pragma warning( disable : 4996 )
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma warning( pop )

REX_NS_BEGIN

// create image w/ width and height
Image::Image( uint16 width, uint16 height )
    : _width( width ),
      _height( height )
{
}

// destroy image
Image::~Image()
{
    uint16* p = const_cast<uint16*>( &_width );
    *p = 0;

    p = const_cast<uint16*>( &_height );
    *p = 0;
}

// get image width
uint16 Image::GetWidth() const
{
    return _width;
}

// get image height
uint16 Image::GetHeight() const
{
    return _height;
}

// save image
bool Image::Save( const char* fname ) const
{
    // convert our floating-point colors into 8-bit colors
    std::vector<uint8> converted;
    converted.resize( _width * _height * 3 );
    size_t convInd = 0;
    for ( auto iter = _hostPixels.begin(); iter != _hostPixels.end(); ++iter, convInd += 3 )
    {
        Color c = *iter;
        converted[ convInd + 0 ] = static_cast<uint8>( Math::Clamp( c.R, 0.0f, 1.0f ) * 255 );
        converted[ convInd + 1 ] = static_cast<uint8>( Math::Clamp( c.G, 0.0f, 1.0f ) * 255 );
        converted[ convInd + 2 ] = static_cast<uint8>( Math::Clamp( c.B, 0.0f, 1.0f ) * 255 );
    }

    // now write out the image as a PNG
    return 0 == stbi_write_png( fname, _width, _height, 3, &( converted[ 0 ] ), _width * 3 );
}

// copy host pixels to device
void Image::CopyHostToDevice()
{
    thrust::copy( _devicePixels.begin(), _devicePixels.end(), _hostPixels.begin() );
}

// copy device pixels to host
void Image::CopyDeviceToHost()
{
}

// set host pixel w/ bounds checking
void Image::SetHostPixel( uint16 x, uint16 y, const Color& color )
{
}

// set host pixel w/o bounds checking
void Image::SetHostPixelUnchecked( uint16 x, uint16 y, const Color& color )
{
}

// set device pixel w/ bounds checking
void Image::SetDevicePixel( uint16 x, uint16 y, const Color& color )
{
}

// set device pixel w/o bounds checking
void Image::SetDevicePixelUnchecked( uint16 x, uint16 y, const Color& color )
{
}

REX_NS_END