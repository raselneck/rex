#include "Rex.hxx"

int main( int argc, char** argv )
{
    rex::Image image( 256, 256 );
    for ( int y = 0; y < 256; ++y )
    {
        for ( int x = 0; x < 256; ++x )
        {
            real32 r = rex::Math::Lerp( 0.0f, 1.0f, x / 255.0f );
            real32 g = rex::Math::Lerp( 0.0f, 1.0f, ( x * y ) / ( 255.0f * 255.0f ) );
            real32 b = rex::Math::Lerp( 0.0f, 1.0f, y / 255.0f );

            image.SetPixelUnchecked( x, y, r, g, b );
        }
    }

    image.Save( "random.png" );

    return 0;
}