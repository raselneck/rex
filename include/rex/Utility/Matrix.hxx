#ifndef __REX_MATRIX_HXX
#define __REX_MATRIX_HXX

#include "../Config.hxx"

REX_NS_BEGIN

// NOTE : M** is M[X-1][Y-1]

/// <summary>
/// Defines a matrix.
/// </summary>
struct Matrix
{
    union
    {
        struct
        {
            real64 M11;
            real64 M12;
            real64 M13;
            real64 M14;

            real64 M21;
            real64 M22;
            real64 M23;
            real64 M24;

            real64 M31;
            real64 M32;
            real64 M33;
            real64 M34;

            real64 M41;
            real64 M42;
            real64 M43;
            real64 M44;
        };
        real64 M[ 4 ][ 4 ];
    };

    /// <summary>
    /// Creates a new matrix.
    /// </summary>
    Matrix();

    /// <summary>
    /// Copies another matrix.
    /// </summary>
    Matrix( const Matrix& );

    /// <summary>
    /// Destroys this matrix.
    /// </summary>
    ~Matrix();

    /// <summary>
    /// Sets this matrix to be the identity matrix.
    /// </summary>
    void SetIdentity();

    Matrix& operator=( const Matrix& );

    bool operator==( const Matrix& ) const;
    bool operator!=( const Matrix& ) const;
};

Matrix operator*( const Matrix&, const Matrix& );
Matrix operator*( const Matrix&, real64 );
Matrix operator*( real64, const Matrix& );
Matrix operator/( const Matrix&, real64 );

REX_NS_END

#include "Matrix.inl"
#endif