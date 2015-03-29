#include <rex/Utility/Image.hxx>
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
      _hPixels( nullptr ),
      _dPixels( nullptr )
{
    // ensure the given sizes are okay
    if ( width > 2048 || height > 2048 )
    {
        Logger::Log( "Cannot create an image of size ", _width, "x", _height, " (max size is 2048x2048)." );
        return;
    }


    // create host pixels
    const uint32 arraySize = _width * _height;
    _hPixels = new Color[ arraySize ];

    // fill host pixels with black
    for ( uint32 i = 0; i < arraySize; ++i )
    {
        _hPixels[ i ] = Color::Black();
    }


    // create device pixels
    cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &_dPixels ), arraySize * sizeof( Color ) );
    if ( err == cudaSuccess )
    {
        // try to copy over the color data
        err = cudaMemcpy( _dPixels, _hPixels, arraySize * sizeof( Color ), cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            // erase the memory
            cudaFree( _dPixels );
            _dPixels = nullptr;

            // print out an error
            Logger::Log( "Failed to copy host image data to device." );
        }
    }
    else
    {
        _dPixels = nullptr;
        Logger::Log( "Failed to allocate space for device image." );
    }
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
    // prepare our converted data array
    std::vector<uint8> converted;
    converted.resize( _width * _height * 3 );

    // convert 32-bit floating point color components into 8-bit components
    const uint32 count = _width * _height;
    for ( uint32 i = 0; i < count; ++i )
    {
        Color& c = _hPixels[ i ];
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
    uint32 size = _width * _height * sizeof( Color );
    cudaMemcpy( _dPixels, _hPixels, size, cudaMemcpyHostToDevice );
}

// copy device pixels to host
void Image::CopyDeviceToHost()
{
    uint32 size = _width * _height * sizeof( Color );
    cudaMemcpy( _hPixels, _dPixels, size, cudaMemcpyDeviceToHost );
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