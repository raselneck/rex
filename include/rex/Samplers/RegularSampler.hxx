#ifndef __REX_REGULARSAMPLER_HXX
#define __REX_REGULARSAMPLER_HXX

#include "../Config.hxx"
#include "Sampler.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a regular sampler.
/// </summary>
class RegularSampler : public Sampler
{
protected:
    /// <summary>
    /// Generates the samples for this regular sampler.
    /// </summary>
    void GenerateSamples();

public:
    /// <summary>
    /// Creates a new regular sampler.
    /// </summary>
    RegularSampler();

    /// <summary>
    /// Creates a new regular sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    RegularSampler( int32 samples );

    /// <summary>
    /// Creates a new regular sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    /// <param name="sets">The initial set count.</param>
    RegularSampler( int32 samples, int32 sets );

    /// <summary>
    /// Destroys this regular sampler.
    /// </summary>
    virtual ~RegularSampler();
};

REX_NS_END

#endif