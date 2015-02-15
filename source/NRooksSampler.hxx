#ifndef __REX_NROOKSSAMPLER_HXX
#define __REX_NROOKSSAMPLER_HXX

#include "Config.hxx"
#include "Sampler.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines an n-Rooks sampler.
/// </summary>
class NRooksSampler : public Sampler
{
    friend class Scene;

protected:
    /// <summary>
    /// Generates the samples for this n-Rooks sampler.
    /// </summary>
    void GenerateSamples();

    /// <summary>
    /// Shuffles the X coordinates used for sampling.
    /// </summary>
    void ShuffleXCoordinates();

    /// <summary>
    /// Shuffles the Y coordinates used for sampling.
    /// </summary>
    void ShuffleYCoordinates();

public:
    /// <summary>
    /// Creates a new n-Rooks sampler.
    /// </summary>
    NRooksSampler();

    /// <summary>
    /// Creates a new n-Rooks sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    NRooksSampler( int32 samples );

    /// <summary>
    /// Creates a new n-Rooks sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    /// <param name="sets">The initial set count.</param>
    NRooksSampler( int32 samples, int32 sets );

    /// <summary>
    /// Destroys this n-Rooks sampler.
    /// </summary>
    virtual ~NRooksSampler();
};

REX_NS_END

#endif