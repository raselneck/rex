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

    std::vector<uchar4> _hPixels;
    uchar4* _dPixels;
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
    /// Gets this image's device memory.
    /// </summary>
    __host__ uchar4* GetDeviceMemory();
};

REX_NS_END