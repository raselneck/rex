#pragma once

#include "../Config.hxx"
#include "../Graphics/Color.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines an image.
/// </summary>
class Image
{
    REX_NONCOPYABLE_CLASS( Image )

    std::vector<Color> _hPixels;
    Color* _dPixels;
    const uint16 _width;
    const uint16 _height;

public:
    /// <summary>
    /// Creates a new image.
    /// </summary>
    /// <param name="width">The image's width.</param>
    /// <param name="height">The image's height.</param>
    __host__ Image( uint16 width, uint16 height );

    /// <summary>
    /// Destroys this image.
    /// </summary>
    __host__ ~Image();

    /// <summary>
    /// Gets this image's width.
    /// </summary>
    __both__ uint16 GetWidth() const;

    /// <summary>
    /// Gets this image's height.
    /// </summary>
    __both__ uint16 GetHeight() const;

    /// <summary>
    /// Saves this image as a PNG to the given file.
    /// </summary>
    /// <param name="fname">The file name.</param>
    __host__ bool Save( const char* fname ) const;

    /// <summary>
    /// Copies the host pixels to the device.
    /// </summary>
    __host__ void CopyHostToDevice();

    /// <summary>
    /// Copies the device pixels to the host.
    /// </summary>
    __host__ void CopyDeviceToHost();

    /// <summary>
    /// Sets the host pixel at the given coordinates with bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="color">The new color.</param>
    __host__ void SetHostPixel( uint16 x, uint16 y, const Color& color );

    /// <summary>
    /// Sets the host pixel at the given coordinates without bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="color">The new color.</param>
    __host__ void SetHostPixelUnchecked( uint16 x, uint16 y, const Color& color );

    /// <summary>
    /// Sets the host pixel at the given coordinates with bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="color">The new color.</param>
    __device__ void SetDevicePixel( uint16 x, uint16 y, const Color& color );

    /// <summary>
    /// Sets the host pixel at the given coordinates without bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="color">The new color.</param>
    __device__ void SetDevicePixelUnchecked( uint16 x, uint16 y, const Color& color );
};

REX_NS_END