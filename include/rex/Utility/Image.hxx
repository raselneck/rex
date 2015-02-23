#ifndef __REX_IMAGE_HXX
#define __REX_IMAGE_HXX

#include "../Config.hxx"
#include "Color.hxx"
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines an image.
/// </summary>
class Image
{
    std::vector<Color> _pixels;
    const uint16 _width;
    const uint16 _height;

public:
    /// <summary>
    /// Creates a new image.
    /// </summary>
    /// <param name="width">The image's width.</param>
    /// <param name="height">The image's height.</param>
    Image( uint16 width, uint16 height );

    /// <summary>
    /// Destroys this image.
    /// </summary>
    ~Image();

    /// <summary>
    /// Gets the pointer to the pixels in this image.
    /// </summary>
    const Color* GetPixels() const;

    /// <summary>
    /// Gets the pixel at the given coordinates, with bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    const Color* GetPixel( uint32 x, uint32 y ) const;

    /// <summary>
    /// Gets the pixel at the given coordinates, without bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    const Color* GetPixelUnchecked( uint32 x, uint32 y ) const;

    /// <summary>
    /// Gets this image's width.
    /// </summary>
    uint16 GetWidth() const;

    /// <summary>
    /// Gets this image's height.
    /// </summary>
    uint16 GetHeight() const;

    /// <summary>
    /// Saves this image to the given file.
    /// </summary>
    /// <param name="fname">The file.</param>
    bool Save( const char* fname ) const;

    /// <summary>
    /// Gets the pointer to the pixels in this image.
    /// </summary>
    Color* GetPixels();

    /// <summary>
    /// Gets the pixel at the given coordinates, with bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    Color* GetPixel( uint32 x, uint32 y );

    /// <summary>
    /// Gets the pixel at the given coordinates, without bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    Color* GetPixelUnchecked( uint32 x, uint32 y );

    /// <summary>
    /// Sets the pixel at the given coordinates, with bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="color">The new color.</param>
    void SetPixel( uint32 x, uint32 y, const Color& color );

    /// <summary>
    /// Sets the pixel at the given coordinates, without bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="color">The new color.</param>
    void SetPixelUnchecked( uint32 x, uint32 y, const Color& color );

    /// <summary>
    /// Sets the pixel at the given coordinates, with bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="r">The red component.</param>
    /// <param name="g">The green component.</param>
    /// <param name="b">The blue component.</param>
    void SetPixel( uint32 x, uint32 y, real32 r, real32 g, real32 b );

    /// <summary>
    /// Sets the pixel at the given coordinates, without bounds checking.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="r">The red component.</param>
    /// <param name="g">The green component.</param>
    /// <param name="b">The blue component.</param>
    void SetPixelUnchecked( uint32 x, uint32 y, real32 r, real32 g, real32 b );
};

REX_NS_END

#endif