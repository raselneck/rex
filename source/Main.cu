#include <stdio.h>
#include <rex/Rex.hxx>
#include <thread>

using namespace rex;

/// <summary>
/// Adds colors with CUDA!
/// </summary>
/// <param name="out">The color array to output to.</param>
/// <param name="lhs">The "left hand side" colors.</param>
/// <param name="rhs">The "right hand side" colors.</param>
/// <param name="colorCount">The total number of colors.</param>
cudaError_t AddColorsWithCuda( Color* out, const Color* lhs, const real32* rhs, uint32 colorCount );

/// <summary>
/// The CUDA kernel for adding two colors.
/// </summary>
/// <param name="out">The color array to output to.</param>
/// <param name="lhs">The "left hand side" colors.</param>
/// <param name="rhs">The "right hand side" colors.</param>
__global__ void KernalAddColor( Color* out, const Color* lhs, const real32* rhs )
{
    int32 index = threadIdx.x;
    
    out[ index ] = rhs[ index ] * lhs[ index ];
}

int32 main( int32 argc, char** argv )
{
    const uint32 arraySize = 10;
    const Color lhs[ arraySize ] =
    {
        Color( 0.1f ),
        Color( 0.2f ),
        Color( 0.3f ),
        Color( 0.4f ),
        Color( 0.5f ),
        Color( 0.6f ),
        Color( 0.7f ),
        Color( 0.8f ),
        Color( 0.9f ),
        Color( 1.0f )
    };
    const real32 rhs[ arraySize ] =
    {
        0.01f,
        0.02f,
        0.03f,
        0.04f,
        0.05f,
        0.06f,
        0.07f,
        0.08f,
        0.09f,
        0.10f
    };
    Color out[ arraySize ];


    // Add vectors in parallel.
    cudaError_t cudaStatus = AddColorsWithCuda( out, lhs, rhs, arraySize );
    if ( cudaStatus != cudaSuccess )
    {
        puts( "AddColorsWithCuda failed!" );
        return 1;
    }


    // print out the colors
    for ( uint32 i = 0; i < arraySize; ++i )
    {
        const Color&  l = lhs[ i ];
        const real32& r = rhs[ i ];
        const Color&  o = out[ i ];

        printf( "{%g, %g, %g} * %g = {%g, %g, %g}\n",
                l.R, l.G, l.B,
                r,
                o.R, o.G, o.B );
    }


    // reset the device to make graphics debugging tools happy
    cudaStatus = cudaDeviceReset();
    if ( cudaStatus != cudaSuccess )
    {
        puts( "cudaDeviceReset failed!" );
        return 1;
    }


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t AddColorsWithCuda( Color* out, const Color* lhs, const real32* rhs, uint32 colorCount )
{
    Color*  devLhs = 0;
    real32* devRhs = 0;
    Color*  devOut = 0;
    cudaError_t status;

    // select the main GPU
    status = cudaSetDevice( 0 );
    if ( status != cudaSuccess )
    {
        puts( "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" );
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    status = cudaMalloc( (void**)&devOut, colorCount * sizeof( Color ) );
    if ( status != cudaSuccess )
    {
        puts( "cudaMalloc failed!" );
        goto Error;
    }

    status = cudaMalloc( (void**)&devLhs, colorCount * sizeof( Color ) );
    if ( status != cudaSuccess )
    {
        puts( "cudaMalloc failed!" );
        goto Error;
    }

    status = cudaMalloc( (void**)&devRhs, colorCount * sizeof( real32 ) );
    if ( status != cudaSuccess )
    {
        puts( "cudaMalloc failed!" );
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    status = cudaMemcpy( devLhs, lhs, colorCount * sizeof( Color ), cudaMemcpyHostToDevice );
    if ( status != cudaSuccess )
    {
        puts( "cudaMemcpy failed!" );
        goto Error;
    }

    status = cudaMemcpy( devRhs, rhs, colorCount * sizeof( real32 ), cudaMemcpyHostToDevice );
    if ( status != cudaSuccess )
    {
        puts( "cudaMemcpy failed!" );
        goto Error;
    }

    // run our kernel with 1 thread block and one thread per color
    KernalAddColor<<<1, colorCount>>>( devOut, devLhs, devRhs );

    // ensure our kernel ran properly
    status = cudaGetLastError();
    if ( status != cudaSuccess )
    {
        printf( "KernalAddColor launch failed: %s\n", cudaGetErrorString( status ) );
        goto Error;
    }

    // wait for the kernel to finish then check for any errors that occurred while running
    status = cudaDeviceSynchronize();
    if ( status != cudaSuccess )
    {
        printf( "cudaDeviceSynchronize returned error code %d after launching KernalAddColor!\n", status );
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    status = cudaMemcpy( out, devOut, colorCount * sizeof( Color ), cudaMemcpyDeviceToHost );
    if ( status != cudaSuccess )
    {
        puts( "cudaMemcpy failed!" );
        goto Error;
    }

Error:
    cudaFree( devOut );
    cudaFree( devLhs );
    cudaFree( devRhs );

    return status;
}
