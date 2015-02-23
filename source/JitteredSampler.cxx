#include <rex/Samplers/JitteredSampler.hxx>
#include <rex/Utility/Random.hxx>

REX_NS_BEGIN

// new jittered sampler
JitteredSampler::JitteredSampler()
    : Sampler()
{
}

// new jittered sampler w/ samples
JitteredSampler::JitteredSampler( int32 samples )
    : Sampler( samples )
{
}

// new jittered sampler w/ samples and sets
JitteredSampler::JitteredSampler( int32 samples, int32 sets )
    : Sampler( samples, sets )
{
}

// destroy jittered sampler
JitteredSampler::~JitteredSampler()
{
}

// generate samples
void JitteredSampler::GenerateSamples()
{
    const int32 n = (int32)sqrt( (real32)_sampleCount );

    _unitSquareSamples.clear();

    for ( int32 set = 0; set < _setCount; ++set )
    {
        for ( int32 y = 0; y < n; ++y )
        {
            for ( int32 x = 0; x < n; ++x )
            {
                _unitSquareSamples.push_back( Vector2(
                    ( x + Random::RandReal32() ) / n,
                    ( y + Random::RandReal32() ) / n
                ) );
            }
        }
    }
}

REX_NS_END