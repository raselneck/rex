#include <stdio.h>
#include <rex/Rex.hxx>
#include <thread>

#define cudaCheck(x, code) { cudaError_t __err = (x); if (__err != cudaSuccess) { rex::Logger::Log("'", #x, "' failed with error code ", __err); } code }

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
    cudaCheck( AddColorsWithCuda( out, lhs, rhs, arraySize ), {} );


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
    cudaCheck( cudaDeviceReset(), {} );


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
    cudaCheck( cudaSetDevice( 0 ), {} );

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaCheck( cudaMalloc( (void**)&devOut, colorCount * sizeof( Color ) ),
               status = __err;
               goto Error; );

    cudaCheck( cudaMalloc( (void**)&devLhs, colorCount * sizeof( Color ) ),
               status = __err;
               goto Error; );

    cudaCheck( cudaMalloc( (void**)&devRhs, colorCount * sizeof( real32 ) ),
               status = __err;
               goto Error; );

    // Copy input vectors from host memory to GPU buffers.
    cudaCheck( cudaMemcpy( devLhs, lhs, colorCount * sizeof( Color ), cudaMemcpyHostToDevice ),
               status = __err;
               goto Error; );

    cudaCheck( cudaMemcpy( devRhs, rhs, colorCount * sizeof( real32 ), cudaMemcpyHostToDevice ),
               status = __err;
               goto Error; );

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
    cudaCheck( cudaDeviceSynchronize(),
               status = __err;
               goto Error; );

    // Copy output vector from GPU buffer to host memory.
    cudaCheck( cudaMemcpy( out, devOut, colorCount * sizeof( Color ), cudaMemcpyDeviceToHost ),
               status = __err;
               goto Error; );

Error:
    cudaFree( devOut );
    cudaFree( devLhs );
    cudaFree( devRhs );

    return cudaSuccess;
}
