#pragma once

#include "../Config.hxx"
#include "../Graphics/Color.hxx"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

REX_NS_BEGIN

/// <summary>
/// Defines an image.
/// </summary>
class Image
{
    thrust::host_vector<Color>   _hostPixels;
    thrust::device_vector<Color> _devicePixels;
    const uint16 _width;
    const uint16 _height;

public:
    /// <summary>
    /// Creates a new image.
    /// </summary>
    /// <param name="width">The image's width.</param>
    /// <param name="height">The image's height.</param>
    __both__ Image( uint16 width, uint16 height );

    /// <summary>
    /// Destroys this image.
    /// </summary>
    __both__ ~Image();

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
    __both__ bool Save( const char* fname ) const;

    /// <summary>
    /// Copies the host pixels to the device.
    /// </summary>
    __both__ void CopyHostToDevice();

    /// <summary>
    /// Copies the device pixels to the host.
    /// </summary>
    __both__ void CopyDeviceToHost();

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