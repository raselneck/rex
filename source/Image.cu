#include <rex/Utility/Image.hxx>
#include <rex/Utility/GC.hxx>
#include <rex/Utility/Logger.hxx>
#include <rex/Math/Math.hxx>
#include <vector>

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
      _height( height ),
      _dPixels( nullptr )
{
    // ensure the given sizes are okay
    if ( width > 1024 || height > 1024 )
    {
        REX_DEBUG_LOG( "Cannot create an image of size ", _width, "x", _height, " (max size is 2048x2048)." );
        return;
    }


    // create host pixels
    const uint32 arraySize = _width * _height;
    _hPixels.resize( arraySize );


    // create device pixels
    _dPixels = GC::DeviceAllocArray( arraySize, &_hPixels[ 0 ] );
}

// destroy image
Image::~Image()
{
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
    // prepare our converted data array
    std::vector<uint8> converted;
    converted.resize( _width * _height * 3 );

    // convert 32-bit floating point color components into 8-bit components
    const uint32 count = _width * _height;
    for ( uint32 i = 0; i < count; ++i )
    {
        const Color& c = _hPixels[ i ];
        converted[ i * 3 + 0 ] = static_cast<uint8>( Math::Clamp( c.R, 0.0f, 1.0f ) * 255 );
        converted[ i * 3 + 1 ] = static_cast<uint8>( Math::Clamp( c.G, 0.0f, 1.0f ) * 255 );
        converted[ i * 3 + 2 ] = static_cast<uint8>( Math::Clamp( c.B, 0.0f, 1.0f ) * 255 );
    }

    // now write out the image as a PNG
    return 0 == stbi_write_png( fname, _width, _height, 3, &( converted[ 0 ] ), _width * 3 );
}

// copy host pixels to device
void Image::CopyHostToDevice()
{
    uint32 size = _hPixels.size() * sizeof( Color );
    cudaMemcpy( _dPixels, &_hPixels[ 0 ], size, cudaMemcpyHostToDevice );
}

// copy device pixels to host
void Image::CopyDeviceToHost()
{
    uint32 size = _hPixels.size() * sizeof( Color );
    cudaMemcpy( &_hPixels[ 0 ], _dPixels, size, cudaMemcpyDeviceToHost );
}

// set host pixel w/ bounds checking
void Image::SetHostPixel( uint16 x, uint16 y, const Color& color )
{
    if ( x < _width && y < _height )
    {
        SetHostPixelUnchecked( x, y, color );
    }
}

// set host pixel w/o bounds checking
void Image::SetHostPixelUnchecked( uint16 x, uint16 y, const Color& color )
{
    _hPixels[ x + y * _width ] = color;
}

// set device pixel w/ bounds checking
__device__ void Image::SetDevicePixel( uint16 x, uint16 y, const Color& color )
{
    if ( x < _width && y < _height )
    {
        SetDevicePixelUnchecked( x, y, color );
    }
}

// set device pixel w/o bounds checking
__device__ void Image::SetDevicePixelUnchecked( uint16 x, uint16 y, const Color& color )
{
    _dPixels[ x + y * _width ] = color;
}

REX_NS_END