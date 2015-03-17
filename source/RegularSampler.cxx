#include <rex/Samplers/RegularSampler.hxx>

REX_NS_BEGIN

// new regular sampler
RegularSampler::RegularSampler()
{
}

// new regular sampler w/ samples
RegularSampler::RegularSampler( int32 samples )
    : Sampler( samples )
{
}

// new regular sampler w/ samples and sets
RegularSampler::RegularSampler( int32 samples, int32 sets )
    : Sampler( samples, sets )
{
}

// destroy sampler
RegularSampler::~RegularSampler()
{
}

// generates samples
void RegularSampler::GenerateSamples()
{
    const int32 n = (int32)sqrtf( (real32)_sampleCount );

    _samples.clear();

    for ( int32 set = 0; set < _setCount; ++set )
    {
        for ( int32 y = 0; y < n; ++y )
        {
            for ( int32 x = 0; x < n; ++x )
            {
                _samples.push_back( Vector2(
                    ( x + 0.5 ) / n, ( y + 0.5 ) / n
                ) );
            }
        }

    }
}

REX_NS_END