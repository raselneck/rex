#include <rex/Math/Matrix.hxx>

REX_NS_BEGIN

// new matrix
Matrix::Matrix()
{
    SetIdentity();
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

// set matrix as identity
void Matrix::SetIdentity()
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

// check for matrix equality
bool Matrix::operator==( const Matrix& m ) const
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

// check for matrix inequality
bool Matrix::operator!=( const Matrix& m ) const
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

// multiply matrix by matrix
Matrix operator*( const Matrix& l, const Matrix& r )
{
    Matrix 	product;

    for ( int y = 0; y < 4; y++ )
    {
        for ( int x = 0; x < 4; x++ )
        {
            real_t sum = 0.0;

            for ( int j = 0; j < 4; j++ )
            {
                sum += l.M[ x ][ j ] * r.M[ j ][ y ];
            }

            product.M[ x ][ y ] = sum;
        }
    }

    return product;
}

// multiply matrix by scalar
Matrix operator*( const Matrix& m, real_t s )
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

// multiply matrix by scalar
Matrix operator*( real_t s, const Matrix& m )
{
    return m * s;
}

// divide matrix by scalar
Matrix operator/( const Matrix& m, real_t s )
{
    real_t invs = 1.0f / s;
    return m * invs;
}

REX_NS_END