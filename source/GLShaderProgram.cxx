#include <rex/GL/GLShaderProgram.hxx>

REX_NS_BEGIN

// create program data
GLShaderProgram::ProgramData::ProgramData()
    : GLHandle      ( 0 )
    , VertexShader  ( GLShaderType::Vertex )
    , FragmentShader( GLShaderType::Fragment )
{
    GLHandle = glCreateProgram();
    glAttachShader( GLHandle, VertexShader.GetHandle() );
    glAttachShader( GLHandle, FragmentShader.GetHandle() );
}

// destroy program data
GLShaderProgram::ProgramData::~ProgramData()
{
    glDetachShader( GLHandle, VertexShader.GetHandle() );
    glDetachShader( GLHandle, FragmentShader.GetHandle() );
    glDeleteProgram( GLHandle );
}



// create program
GLShaderProgram::GLShaderProgram()
    : _handle( new ProgramData() )
{
}

// destroy program
GLShaderProgram::~GLShaderProgram()
{
}

// use program
void GLShaderProgram::Use() const
{
    glUseProgram( _handle->GLHandle );
}

// attempt to link program
bool GLShaderProgram::Link() const
{
    glLinkProgram( _handle->GLHandle );

    // check link status
    GLint linkStatus = 0;
    glGetProgramiv( _handle->GLHandle, GL_LINK_STATUS, &linkStatus );
    if ( linkStatus == GL_TRUE )
    {
        return true;
    }

    return false;
}

// get information log
String GLShaderProgram::GetLog() const
{
    // get log length
    GLint logLength = 0;
    glGetProgramiv( _handle->GLHandle, GL_INFO_LOG_LENGTH, &logLength );

    // get log
    String log;
    log.resize( size_t( logLength ) );
    glGetProgramInfoLog( _handle->GLHandle, logLength, NULL, &log[ 0 ] );

    return log;
}

// find uniform location
int GLShaderProgram::FindUniformLocation( const char* name ) const
{
    return glGetUniformLocation( _handle->GLHandle, name );
}

// get vertex shader
GLShader& GLShaderProgram::GetVertexShader()
{
    return _handle->VertexShader;
}

// get fragment shader
GLShader& GLShaderProgram::GetFragmentShader()
{
    return _handle->FragmentShader;
}

// set texture
void GLShaderProgram::SetTexture( int index, const GLTexture2D& texture, int unit )
{
    glActiveTexture( GL_TEXTURE0 + unit );
    glBindTexture( GL_TEXTURE_2D, texture.GetOpenGLHandle() );
    glProgramUniform1i( _handle->GLHandle, index, unit );
}

REX_NS_END