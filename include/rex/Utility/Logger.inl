#pragma once

#include <iostream>
#include <sstream>

// Log method adapted from here: http://stackoverflow.com/a/25386444

REX_NS_BEGIN

#pragma warning( push )
#pragma warning( disable : 4789 )

// convert item to string
template<typename T> inline String Logger::ToString( const T& value )
{
    std::ostringstream stream;
    stream.precision( 16 );
    stream << value;
    return stream.str();
}

// merge list of strings
inline String Logger::Merge( std::initializer_list<String> list )
{
    std::ostringstream stream;
    for ( const String& s : list )
    {
        stream << s;
    }
    return stream.str();
}

// log arguments to the console
template<typename ... Args> inline void Logger::Log( const Args& ... args )
{
    std::lock_guard<std::mutex> lock( _mutex );

    std::cout << Merge( { ToString( args )... } ) << std::endl;
}

#pragma warning( pop )

REX_NS_END