#ifndef __REX_MULTIJITTEREDSAMPLER_HXX
#define __REX_MULTIJITTEREDSAMPLER_HXX

#include "../Config.hxx"
#include "Sampler.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a multi-jittered sampler.
/// </summary>
class MultiJitteredSampler : public Sampler
{
protected:
    /// <summary>
    /// Generates the samples for this multi-jittered sampler.
    /// </summary>
    void GenerateSamples();

public:
    /// <summary>
    /// Creates a new multi-jittered sampler.
    /// </summary>
    MultiJitteredSampler();

    /// <summary>
    /// Creates a new multi-jittered sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    MultiJitteredSampler( int32 samples );

    /// <summary>
    /// Creates a new multi-jittered sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    /// <param name="sets">The initial set count.</param>
    MultiJitteredSampler( int32 samples, int32 sets );

    /// <summary>
    /// Destroys this multi-jittered sampler.
    /// </summary>
    virtual ~MultiJitteredSampler();
};

REX_NS_END

#endif