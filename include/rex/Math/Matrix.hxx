#pragma once

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
            real_t M11, M12, M13, M14;
            real_t M21, M22, M23, M24;
            real_t M31, M32, M33, M34;
            real_t M41, M42, M43, M44;
        };
        real_t M[ 4 ][ 4 ];
    };

    /// <summary>
    /// Creates a new matrix.
    /// </summary>
    __both__ Matrix();

    /// <summary>
    /// Destroys this matrix.
    /// </summary>
    __both__ ~Matrix();

    /// <summary>
    /// Sets this matrix to be the identity matrix.
    /// </summary>
    __both__ void SetIdentity();

    __both__ bool operator==( const Matrix& ) const;
    __both__ bool operator!=( const Matrix& ) const;
};

__both__ Matrix operator*( const Matrix&, const Matrix& );
__both__ Matrix operator*( const Matrix&, real_t );
__both__ Matrix operator*( real_t, const Matrix& );
__both__ Matrix operator/( const Matrix&, real_t );

REX_NS_END