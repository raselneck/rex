#ifndef __REX_SAMPLER_HXX
#define __REX_SAMPLER_HXX

#include "../Config.hxx"
#include "../Utility/Vector2.hxx"
#include "../Utility/Vector3.hxx"
#include <mutex>
#include <vector>

REX_NS_BEGIN

/// <summary>
/// Defines the base for all samplers.
/// </summary>
class Sampler
{
    friend class Scene;

    // having _indexJump and _samplePointCount be mutable
    // allows us to have the sample methods be const

protected:
    std::vector<Vector2> _samples;
    std::vector<int32>   _indices;
    mutable std::mutex   _mutex;
    mutable int32        _indexJump;
    mutable uint32       _samplePointCount;
    int32                _sampleCount;
    int32                _setCount;

    /// <summary>
    /// Sets up the randomly shuffled indices.
    /// </summary>
    void SetupShuffledIndices();
    
    /// <summary>
    /// Generates the samples in the unit square.
    /// <summary>
    virtual void GenerateSamples() = 0;

public:
    /// <summary>
    /// Creates a new sampler.
    /// </summary>
    Sampler();

    /// <summary>
    /// Creates a new sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    Sampler( int32 samples );

    /// <summary>
    /// Creates a new sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    /// <param name="sets">The initial set count.</param>
    Sampler( int32 samples, int32 sets );

    /// <summary>
    /// Destroys this sampler.
    /// </summary>
    virtual ~Sampler();

    /// <summary>
    /// Gets the sample count.
    /// <summary>
    int32 GetSampleCount() const;

    /// <summary>
    /// Gets the set count.
    /// <summary>
    int32 GetSetCount() const;

    /// <summary>
    /// Gets the next sample on the unit square.
    /// </summary>
    Vector2 SampleUnitSquare() const;

    /// <summary>
    /// Sets the sample count.
    /// </summary>
    /// <param name="samples">The new sample count.</param>
    void SetSampleCount( int32 samples );

    /// <summary>
    /// Sets the set count.
    /// </summary>
    /// <param name="sets">The new set count.</param>
    void SetSetCount( int32 sets );
};

REX_NS_END

#endif