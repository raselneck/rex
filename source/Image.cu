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
    // create host pixels
    const uint_t arraySize = _width * _height;
    const uint_t cudaSize  = arraySize * sizeof( uchar4 );
    _hPixels.resize( arraySize );


    // create device pixels
    if ( cudaSuccess != cudaMalloc( reinterpret_cast<void**>( &_dPixels ), cudaSize ) )
    {
        REX_DEBUG_LOG( "Failed to allocate device image pixels." );
        return;
    }
    if ( cudaSuccess != cudaMemcpy( _dPixels, &_hPixels[ 0 ], cudaSize, cudaMemcpyHostToDevice ) )
    {
        REX_DEBUG_LOG( "Failed to copy over initial device pixels." );
        cudaFree( _dPixels );
        _dPixels = nullptr;
    }
}

// destroy image
Image::~Image()
{
    // free our device pixels
    if ( _dPixels )
    {
        cudaFree( _dPixels );
        _dPixels = nullptr;
    }
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
    return 0 == stbi_write_png( fname, _width, _height, 4, &( _hPixels[ 0 ] ), _width * 4 );
}

// copy host pixels to device
void Image::CopyHostToDevice()
{
    uint_t size = _hPixels.size() * sizeof( uchar4 );
    cudaMemcpy( _dPixels, &_hPixels[ 0 ], size, cudaMemcpyHostToDevice );
}

// copy device pixels to host
void Image::CopyDeviceToHost()
{
    uint_t size = _hPixels.size() * sizeof( uchar4 );
    cudaMemcpy( &_hPixels[ 0 ], _dPixels, size, cudaMemcpyDeviceToHost );
}

// get image device memory
uchar4* Image::GetDeviceMemory()
{
    return _dPixels;
}

REX_NS_END