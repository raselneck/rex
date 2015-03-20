#include <iostream>
#include <sstream>

REX_NS_BEGIN

#pragma warning( push )
#pragma warning( disable : 4789 )

// log arguments to the console
template<typename ... Args> void Logger::Log( const Args& ... args )
{
    std::lock_guard<std::mutex> lock( _mutex );

    // adapted from http://stackoverflow.com/a/21812549

    std::stringstream stream;
    stream.precision( 16 );
    using List = int[];
    (void)List { 0, ( (void)( stream << args ), 0 ) ... };

    std::cout << stream.str() << std::endl;
}

#pragma warning( pop )

REX_NS_END