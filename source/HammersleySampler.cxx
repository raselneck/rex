#include <rex/Samplers/HammersleySampler.hxx>
#include <rex/Utility/Random.hxx>

// Hammersley sampler defined in "Ray Tracing from the Ground Up," pages 107-110

REX_NS_BEGIN

// TODO : Suffern explained this as the inverse radical, but that's not what it is...
static real64 GetPhi( int32 i )
{
    // NOTE: this function is called "phi" by Suffern

    real64 x = 0.0;
    real64 f = 0.5;

    while ( i )
    {
        x += f * static_cast<real64>( i % 2 );
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
            vec.Y = GetPhi( i );

            _unitSquareSamples.push_back( vec );
        }
    }
}

REX_NS_END