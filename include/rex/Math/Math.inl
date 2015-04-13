REX_NS_BEGIN

// get pi
inline real_t Math::Pi()
{
    return real_t( 3.14159265358979323846264338327950 );
}

// get 2 * pi
inline real_t Math::TwoPi()
{
    return real_t( 6.28318530717958647692528676655900 );
}

// get pi / 180
inline real_t Math::PiOver180()
{
    return real_t( 0.01745329251994329576923690768489 );
}

// get 1 / pi
inline real_t Math::InvPi()
{
    return real_t( 0.31830988618379067153776752674503 );
}

// get 1 / ( 2 * pi )
inline real_t Math::InvTwoPi()
{
    return real_t( 0.15915494309189533576888376337251 );
}

// get a really small value
inline real_t Math::Epsilon()
{
    return real_t( 0.0001 );
}

// get a huge value
inline real_t Math::HugeValue()
{
    return real_t( 1.0E10 );
}

// get absolute value
template<class T> inline T Math::Abs( T value )
{
    if ( value < T( 0 ) )
    {
        return -value;
    }
    return value;
}

// get minimum
template<class T> inline T Math::Min( T val1, T val2 )
{
    return ( val1 < val2 ) ? val1 : val2;
}

// get maximum
template<class T> inline T Math::Max( T val1, T val2 )
{
    return ( val1 > val2 ) ? val1 : val2;
}

// clamps to range
template<class T> inline T Math::Clamp( T value, T min, T max )
{
    return Min( max, Max( min, value ) );
}

// linearly interpolate value
template<class T> inline T Math::Lerp( T v0, T v1, T t )
{
    return ( T( 1 ) - t ) * v0 + t * v1;
}

// map from one range to another
template<class T> inline T Math::Map( T value, T min1, T max1, T min2, T max2 )
{
    // derived from http://stackoverflow.com/questions/5731863/mapping-a-numeric-range-onto-another

    real64 slope = 1.0 * ( max2 - min2 ) / ( max1 - min1 );
    real64 out = min2 + slope * ( value - min1 );
    return static_cast<T>( Round( out ) );
}

REX_NS_END