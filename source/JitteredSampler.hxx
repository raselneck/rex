#ifndef __REX_JITTEREDSAMPLER_HXX
#define __REX_JITTEREDSAMPLER_HXX

#include "Config.hxx"
#include "Sampler.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a jittered sampler.
/// </summary>
class JitteredSampler : public Sampler
{
    friend class Scene;

protected:
    /// <summary>
    /// Generates the samples for this jittered sampler.
    /// </summary>
    void GenerateSamples();

public:
    /// <summary>
    /// Creates a new jittered sampler.
    /// </summary>
    JitteredSampler();

    /// <summary>
    /// Creates a new jittered sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    JitteredSampler( int32 samples );

    /// <summary>
    /// Creates a new jittered sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    /// <param name="sets">The initial set count.</param>
    JitteredSampler( int32 samples, int32 sets );

    /// <summary>
    /// Destroys this jittered sampler.
    /// </summary>
    virtual ~JitteredSampler();
};

REX_NS_END

#endif