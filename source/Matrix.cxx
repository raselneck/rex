#include <rex/Utility/Matrix.hxx>

REX_NS_BEGIN

// new matrix
Matrix::Matrix()
{
    SetIdentity();
}

// copy matrix
Matrix::Matrix( const Matrix& other )
{
    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            M[ x ][ y ] = other.M[ x ][ y ];
        }
    }
}

// destroy matrix
Matrix::~Matrix()
{
    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            M[ x ][ y ] = 0.0;
        }
    }
}

REX_NS_END