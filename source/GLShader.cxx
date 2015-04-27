#include <rex/GL/GLShader.hxx>
#include <rex/Utility/Logger.hxx>
#include <fstream>
#include <sstream>

REX_NS_BEGIN

// destroys a shader handle
static void DestroyShaderHandle( void* data )
{
    GLuint* handle = reinterpret_cast<GLuint*>( data );

    glDeleteShader( *handle );

    delete handle;
}

// creates a shader handle
static Handle<GLuint> CreateShaderHandle( GLShaderType type )
{
    Handle<GLuint> handle = Handle<GLuint>( new GLuint( 0 ), DestroyShaderHandle );

    *handle = glCreateShader( static_cast<GLenum>( type ) );

    return handle;
}


// create new shader
GLShader::GLShader( GLShaderType type )
    : _type( type )
    , _handle( CreateShaderHandle( type ) )
{
}

// destroy shader
GLShader::~GLShader()
{
}

// get shader handle
GLuint GLShader::GetHandle() const
{
    return *_handle;
}

// get info log
String GLShader::GetLog() const
{
    // get log length
    GLint logLength = 0;
    glGetShaderiv( *_handle, GL_INFO_LOG_LENGTH, &logLength );

    // get the log
    String log;
    log.resize( size_t( logLength ) );
    glGetShaderInfoLog( *_handle, logLength, NULL, &log[ 0 ] );

    return log;
}

// load shader file
bool GLShader::LoadFile( const String& fname )
{
    // open the file
    std::ifstream in( fname.c_str(), std::ios::in | std::ios::binary );
    if ( !in.is_open() )
    {
        REX_DEBUG_LOG( "Failed to open shader file '", fname, "'" );
        return false;
    }

    // read the file
    std::stringstream stream;
    stream << in.rdbuf();
    String source = stream.str();

    // close the file
    in.close();

    // compile the file source
    return LoadSource( source );
}

// load shader source
bool GLShader::LoadSource( const String& source )
{
    // compile shader
    const char* csrc = source.c_str();
    glShaderSource( *_handle, 1, &csrc, NULL );
    glCompileShader( *_handle );

    // check compile status
    GLint compileStatus = 0;
    glGetShaderiv( *_handle, GL_COMPILE_STATUS, &compileStatus );
    if ( compileStatus != GL_TRUE )
    {
        return false;
    }

    return true;
}

REX_NS_END