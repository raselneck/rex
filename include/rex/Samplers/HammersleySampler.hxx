#ifndef __REX_HAMMERSLEYSAMPLER_HXX
#define __REX_HAMMERSLEYSAMPLER_HXX

#include "../Config.hxx"
#include "Sampler.hxx"

REX_NS_BEGIN

/// <summary>
/// Defines a Hammersley sampler.
/// </summary>
class HammersleySampler : public Sampler
{
protected:
    /// <summary>
    /// Generates the samples for this Hammersley sampler.
    /// </summary>
    void GenerateSamples();

public:
    /// <summary>
    /// Creates a new Hammersley sampler.
    /// </summary>
    HammersleySampler();

    /// <summary>
    /// Creates a new Hammersley sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    HammersleySampler( int32 samples );

    /// <summary>
    /// Creates a new Hammersley sampler.
    /// </summary>
    /// <param name="samples">The initial sample count.</param>
    /// <param name="sets">The initial set count.</param>
    HammersleySampler( int32 samples, int32 sets );

    /// <summary>
    /// Destroys this Hammersley sampler.
    /// </summary>
    virtual ~HammersleySampler();
};

REX_NS_END

#endif