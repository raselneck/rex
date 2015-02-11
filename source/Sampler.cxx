#include "Sampler.hxx"
#include "Random.hxx"
#include <algorithm> // for std::random_shuffle

REX_NS_BEGIN

// create new sampler
Sampler::Sampler()
    : Sampler( REX_DEFAULT_SAMPLES, REX_DEFAULT_SETS )
{
}

// new sampler w/ samples
Sampler::Sampler( int32 samples )
    : Sampler( 1, REX_DEFAULT_SETS )
{
}

// new sampler w/ samples and sets
Sampler::Sampler( int32 samples, int32 sets )
    : _sampleCount( samples ),
      _setCount( sets ),
      _indexJump( 0 ),
      _samplePointCount( 0 )
{
    
    SetupShuffledIndices();
}

// destroy sampler
Sampler::~Sampler()
{
    _sampleCount      = 0;
    _setCount         = 0;
    _indexJump        = 0;
    _samplePointCount = 0;
}

// get sample count
int32 Sampler::GetSampleCount() const
{
    return _sampleCount;
}

// get set count
int32 Sampler::GetSetCount() const
{
    return _setCount;
}

// setup shuffled indices
void Sampler::SetupShuffledIndices()
{
    // clear any existing data
    _unitSquareSamples.clear();
    _indices.clear();

    // reserve memory and create our temporary index buffer
    _unitSquareSamples.reserve( _sampleCount * _setCount );
    _indices.reserve( _sampleCount * _setCount );
    std::vector<int32> temp;

    // populate the temporary index buffer with our sample indices
    for ( int32 i = 0; i < _sampleCount; ++i )
    {
        temp.push_back( i );
    }

    // now create our randomly shuffled indices
    for ( int32 i = 0; i < _setCount; ++i )
    {
        std::random_shuffle( temp.begin(), temp.end() );

        for ( int32 j = 0; j < _sampleCount; ++j )
        {
            _indices.push_back( temp[ j ] );
        }
    }
    
}

// get next sample on unit square
Vector2 Sampler::SampleUnitSquare() const
{
    // check if we're on a new pixel
    if ( _samplePointCount % _sampleCount == 0 )
    {
        _indexJump = ( Random::RandInt() % _setCount ) * _sampleCount;
    }

    // get the sample index then the sample
    int32 index = _indices[ _indexJump + _samplePointCount % _sampleCount ];
    ++_samplePointCount;
    return _unitSquareSamples[ _indexJump + index ];
}

// sample unit disk
Vector2 Sampler::SampleUnitDisk() const
{
    throw "Not implemented.";
}

// sample unit hemisphere
Vector3 Sampler::SampleUnitHemisphere() const
{
    throw "Not implemented.";
}

// sample unit sphere
Vector3 Sampler::SampleUnitSphere() const
{
    throw "Not implemented.";
}

// set sampler count
void Sampler::SetSampleCount( int32 samples )
{
    _sampleCount = samples;

    SetupShuffledIndices();
    GenerateSamples();
}

// set set count
void Sampler::SetSetCount( int32 sets )
{
    _setCount = sets;

    SetupShuffledIndices();
    GenerateSamples();
}

REX_NS_END