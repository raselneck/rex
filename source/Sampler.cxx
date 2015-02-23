#include <rex/Samplers/Sampler.hxx>
#include <rex/Utility/Math.hxx>
#include <rex/Utility/Random.hxx>
#include <algorithm> // for std::random_shuffle
#include <exception> // TODO : TEMPORARY!!

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
        _indexJump = ( Random::RandInt32() % _setCount ) * _sampleCount;
    }

    // get the sample index then the sample
    int32 index = _indices[ _indexJump + _samplePointCount % _sampleCount ];
    ++_samplePointCount;
    return _unitSquareSamples[ _indexJump + index ];
}

// sample unit disk
Vector2 Sampler::SampleUnitDisk() const
{
    // check if we're on a new pixel
    if ( _samplePointCount % _sampleCount == 0 )
    {
        _indexJump = ( Random::RandInt32() % _setCount ) * _sampleCount;
    }

    // get the sample index then the sample
    int32 index = _indices[ _indexJump + _samplePointCount % _sampleCount ];
    ++_samplePointCount;
    return _unitDiskSamples[ _indexJump + index ];
}

// sample unit hemisphere
Vector3 Sampler::SampleUnitHemisphere() const
{
    // check if we're on a new pixel
    if ( _samplePointCount % _sampleCount == 0 )
    {
        _indexJump = ( Random::RandInt32() % _setCount ) * _sampleCount;
    }

    // get the sample index then the sample
    int32 index = _indices[ _indexJump + _samplePointCount % _sampleCount ];
    ++_samplePointCount;
    return _unitHemisphereSamples[ _indexJump + index ];
}

// sample unit sphere
Vector3 Sampler::SampleUnitSphere() const
{
    throw std::exception( "Not implemented." );
}

// map samples to unit disk
void Sampler::MapSamplesToUnitDisk()
{
    // adapted from Suffern, 123

    const int32 size = static_cast<int32>( _unitSquareSamples.size() );
    real64 r, phi;      // polar coordinates
    Vector2 sp;         // sample point

    // reserve space for new samples
    _unitDiskSamples.clear();
    _unitDiskSamples.reserve( size );

    // now we need to convert unit square coordinates to polar coordinates
    for ( int32 i = 0; i < size; ++i )
    {
        // map sample point to [-1, 1]x[-1, 1]
        sp.X = 2.0 * _unitDiskSamples[ i ].X - 1.0;
        sp.Y = 2.0 * _unitDiskSamples[ i ].Y - 1.0;

        // now we need to check which quadrant we're in
        if ( sp.X > -sp.Y )
        {
            if ( sp.X > sp.Y )
            {
                // quadrant 1
                r   = sp.X;
                phi = sp.Y / sp.X;
            }
            else
            {
                // quadrant 2
                r   = sp.Y;
                phi = 2.0 - sp.X / sp.Y;
            }
        }
        else
        {
            if ( sp.X < sp.Y )
            {
                // quadrant 3
                r   = -sp.X;
                phi = 4.0 + sp.Y / sp.X;
            }
            else
            {
                // quadrant 4
                r = -sp.Y;
                phi = ( sp.Y == 0.0 )
                    ? 0.0
                    : 6.0 - sp.X / sp.Y;
            }
        }

        // now create the polar coordinate
        phi *= Math::PI * 0.25;
        _unitDiskSamples[ i ].X = r * cos( phi );
        _unitDiskSamples[ i ].Y = r * sin( phi );
    }

    // erase the unit square samples to save on memory
    _unitSquareSamples.erase( _unitDiskSamples.begin(), _unitDiskSamples.end() );
}

// map samples to unit hemisphere
void Sampler::MapSamplesToUnitHemisphere( real32 exponent )
{
    // adapted from Suffern, 129

    const int32 size = static_cast<int32>( _unitSquareSamples.size() );

    // reserve memory
    _unitHemisphereSamples.clear();
    _unitHemisphereSamples.reserve( _sampleCount * _setCount );

    // convert from 2D Cartesian to 3D half-spherical
    Vector3 vec;
    for ( int32 i = 0; i < size; i++ )
    {
        real64 cosPhi   = cos( 2.0 * Math::PI * _unitSquareSamples[ i ].X );
        real64 sinPhi   = sin( 2.0 * Math::PI * _unitSquareSamples[ i ].X );
        real64 cosTheta = pow( ( 1.0 - _unitSquareSamples[ i ].Y ), 1.0 / ( exponent + 1.0 ) );
        real64 sinTheta = sqrt( 1.0 - cosTheta * cosTheta );
        vec.X = sinTheta * cosPhi;
        vec.Y = sinTheta * sinPhi;
        vec.Z = cosTheta;
        _unitHemisphereSamples.push_back( vec );
    }
}

// map samples to unit sphere
void Sampler::MapSamplesToUnitSphere()
{
    throw std::exception( "Not implemented." );
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