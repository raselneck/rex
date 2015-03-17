#ifndef __REX_DEBUG_HXX
#define __REX_DEBUG_HXX

#pragma warning( push )
#pragma warning( disable : 4789 )

#include "Config.hxx"
#include "Utility/Vector2.hxx"
#include "Utility/Vector3.hxx"
#include "Geometry/BoundingBox.hxx"
#include <iostream>
#include <sstream>

#if defined( _MSC_VER )

/// <summary>
/// Custom assert macro for Rex.
/// </summary>
/// <param name="cond">The condition to check.</param>
/// <param name="message">The message to log if the condition fails.</param>
#define REX_ASSERT(cond, message) { \
        bool __result = static_cast<bool>( cond ); \
        if ( !__result ) { \
            rex::String __fname = __FILE__; \
            __fname = __fname.substr( __fname.find_last_of( '\\' ) + 1 ); \
            rex::WriteLine( "(", __fname, ":", __LINE__, ") Assertion failed: ", message, "\n  ", #cond, "\n" ); \
            __debugbreak(); /* NOTE : Visual Studio only! */ \
                } \
        }

#else

#include <assert.h>
#define REX_ASSERT(cond, message) assert((cond) && message)

#endif

REX_NS_BEGIN

/// <summary>
/// Logs the given arguments.
/// </summary>
/// <param name="args">The arguments to log.</param>
template<typename ... Args> inline void Write( const Args& ... args )
{
    std::stringstream stream;

    using List = int[];
    (void)List{ 0, ( (void)( stream << args ), 0 ) ... };

    std::cout << stream.str();
}

/// <summary>
/// Logs the given arguments.
/// </summary>
/// <param name="args">The arguments to log.</param>
template<typename ... Args> inline void WriteLine( const Args& ... args )
{
    std::stringstream stream;

    using List = int[];
    (void)List{ 0, ( (void)( stream << args ), 0 ) ... };

    std::cout << stream.str() << std::endl;
}

/// <summary>
/// Reads a line of text from the console.
/// </summary>
inline String ReadLine()
{
    String str;
    std::getline( std::cin, str );
    return str;
}

REX_NS_END

// print Vector2
static std::ostream& operator<<( std::ostream& stream, const rex::Vector2 vec )
{
    stream.precision( 12 );
    stream << "{" << vec.X << ", " << vec.Y << "}";

    return stream;
}

// print Vector3
static std::ostream& operator<<( std::ostream& stream, const rex::Vector3 vec )
{
    stream.precision( 12 );
    stream << "{" << vec.X << ", " << vec.Y << ", " << vec.Z << "}";

    return stream;
}

// print BoundingBox
static std::ostream& operator<<( std::ostream& stream, const rex::BoundingBox box )
{
    stream << "{" << box.GetMin() << ", " << box.GetMax() << "}";

    return stream;
}

#pragma warning( pop )

#endif