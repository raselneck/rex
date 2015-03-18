REX_NS_BEGIN

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