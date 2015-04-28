#include <iostream>
#include <sstream>
#include "../Math/Math.hxx"

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

// get absolute file name
inline String Logger::GetAbsoluteFileName( const char* fname )
{
    String str = fname;
    uint32 index = Math::Min( str.find_last_of( '\\' ),
                              str.find_last_of( '/'  ) );
    return str.substr( index + 1 );
}

// log arguments to the console
template<typename ... Args> inline void Logger::Log( const Args& ... args )
{
    std::lock_guard<std::mutex> lock( _mutex );

    std::cout << Merge( { ToString( args )... } ) << std::endl;
}

#pragma warning( pop )

REX_NS_END