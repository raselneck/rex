#include "Logger.hxx"
#include <iostream>

REX_NS_BEGIN

std::ofstream Logger::_fout( "rex.log", std::ios::out | std::ios::binary );
std::ostream& Logger::_cout = std::cout;

REX_NS_END