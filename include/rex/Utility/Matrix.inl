#ifndef __REX_MATRIX_INL
#define __REX_MATRIX_INL

#include "Matrix.hxx"

REX_NS_BEGIN

// set matrix as identity
inline void Matrix::SetIdentity()
{
    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            if ( x == y )
            {
                M[ x ][ y ] = 1.0;
            }
            else
            {
                M[ x ][ y ] = 0.0;
            }
        }
    }
}

inline Matrix& Matrix::operator=( const Matrix& other )
{
    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            M[ x ][ y ] = other.M[ x ][ y ];
        }
    }

    return *this;
}

inline bool Matrix::operator==( const Matrix& m ) const
{
    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            if ( M[ x ][ y ] != m.M[ x ][ y ] )
            {
                return false;
            }
        }
    }

    return true;
}

inline bool Matrix::operator!=( const Matrix& m ) const
{
    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            if ( M[ x ][ y ] != m.M[ x ][ y ] )
            {
                return true;
            }
        }
    }

    return false;
}

inline Matrix operator*( const Matrix& l, const Matrix& r )
{
    Matrix 	product;

    for ( int y = 0; y < 4; y++ )
    {
        for ( int x = 0; x < 4; x++ )
        {
            real64 sum = 0.0;

            for ( int j = 0; j < 4; j++ )
            {
                sum += l.M[ x ][ j ] * r.M[ j ][ y ];
            }

            product.M[ x ][ y ] = sum;
        }
    }

    return product;
}

inline Matrix operator*( const Matrix& m, real64 s )
{
    Matrix m2;

    for ( int x = 0; x < 4; ++x )
    {
        for ( int y = 0; y < 4; ++y )
        {
            m2.M[ x ][ y ] = m.M[ x ][ y ] * s;
        }
    }

    return m2;
}

inline Matrix operator*( real64 s, const Matrix& m )
{
    return m * s;
}

inline Matrix operator/( const Matrix& m, real64 s )
{
    real64 invs = 1.0 / s;
    return m * invs;
}

REX_NS_END

#endif