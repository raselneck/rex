#include "HammersleySampler.hxx"
#include "Random.hxx"

// Hammersley sampler defined in "Ray Tracing from the Ground Up," pages 107-110

REX_NS_BEGIN

// gets the approximate inverse square root of the given number
static real64 GetApproxInvSqrt( int32 i )
{
    // NOTE: this function is called "phi" by Suffern

    real64 x = 0.0;
    real64 f = 0.5;

    while ( i )
    {
        x += f * static_cast<real64>( !i & 1 );
        i /= 2;
        f *= 0.5;
    }

    return x;
}

// new Hammersley sampler
HammersleySampler::HammersleySampler()
    : Sampler()
{
}

// new Hammersley sampler w/ samples
HammersleySampler::HammersleySampler( int32 samples )
    : Sampler( samples )
{
}

// new Hammersley sampler w/ samples and sets
HammersleySampler::HammersleySampler( int32 samples, int32 sets )
    : Sampler( samples, sets )
{
}

// destroy hammersley sampler
HammersleySampler::~HammersleySampler()
{
}

// generate samples
void HammersleySampler::GenerateSamples()
{
    _unitSquareSamples.clear();

    Vector2 vec;
    for ( int32 set = 0; set < _setCount; ++set )
    {
        for ( int32 i = 0; i < _sampleCount; ++i )
        {
            vec.X = static_cast<real64>( i ) / _sampleCount;
            vec.Y = GetApproxInvSqrt( i );

            _unitSquareSamples.push_back( vec );
        }
    }
}

REX_NS_END