#include "BRDF.hxx"

REX_NS_BEGIN

// new BRDF
BRDF::BRDF( Handle<Sampler>& sampler )
{
    _sampler = sampler;
}

// destroy BRDF
BRDF::~BRDF()
{
}

REX_NS_END