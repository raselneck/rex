#include <rex/BRDFs/BRDF.hxx>

REX_NS_BEGIN

// create BRDF
BRDF::BRDF()
{
}

// create BRDF w/ sampler
BRDF::BRDF( Handle<Sampler>& sampler )
{
    _sampler = sampler;
}

// destroy BRDF
BRDF::~BRDF()
{
}

// set sampler
void BRDF::SetSampler( Handle<Sampler>& sampler )
{
    _sampler = sampler;
}

REX_NS_END