#include <rex/Graphics/TextureRenderer.hxx>
#include <rex/Utility/Logger.hxx>

REX_NS_BEGIN

// create the texture renderer
TextureRenderer::TextureRenderer( const GLTexture2D* texture )
    : _texture( texture )
    , _uniformTexLocation( 0 )
    , _vao( 0 )
    , _vbo( 0 )
{
    CreateBufferObjects();
    CreateShaderProgram();
}

// destroy the texture renderer
TextureRenderer::~TextureRenderer()
{
    _texture = nullptr;
}

// create the VBO and VAO
void TextureRenderer::CreateBufferObjects()
{
    // create the VAO and VBO
    glGenVertexArrays( 1, &_vao );
    glBindVertexArray( _vao );

    glCreateBuffers( 1, &_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, _vbo );

    // create the buffer data
    GLfloat data[] =
    { //   x      y        u     v  
        -1.0f, -1.0f,    0.0f, 1.0f,
        -1.0f,  1.0f,    0.0f, 0.0f,
         1.0f, -1.0f,    1.0f, 1.0f,
                         
         1.0f,  1.0f,    1.0f, 0.0f,
         1.0f, -1.0f,    1.0f, 1.0f,
        -1.0f,  1.0f,    0.0f, 0.0f
    };

    // upload the buffer data
    glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 24, data, GL_STATIC_DRAW );

    // specify how the data is laid out
    glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, sizeof( GLfloat ) * 4, 0 );                                // X, Y
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, sizeof( GLfloat ) * 4, (void*)( sizeof( GLfloat ) * 2 ) ); // U, V
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );

    // unbind our buffers
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glBindVertexArray( 0 );
}

// create the shader program
void TextureRenderer::CreateShaderProgram()
{
    // define the vertex shader
    static const char* vertexShaderSource = GLSL( 400,
        layout( location = 0 ) in vec2 position;            \n
        layout( location = 1 ) in vec2 vertTexCoords;       \n
        out vec2 texCoords;                                 \n
        void main()                                         \n
        {                                                   \n
            texCoords = vertTexCoords;                      \n
            gl_Position = vec4(position, 0, 1);             \n
        }                                                   \n
    );

    // define the fragment shader
    static const char* fragmentShaderSource = GLSL( 400,
        layout( location = 0 ) out vec4 fragColor;          \n
        in      vec2      texCoords;                        \n
        uniform sampler2D tex;                              \n
        void main()                                         \n
        {                                                   \n
            vec4 texColor = texture( tex, texCoords );      \n
            fragColor = vec4( texColor.rgb, 1 );            \n
        }                                                   \n
    );


    // attempt to compile the vertex shader
    if ( !_program.GetVertexShader().LoadSource( vertexShaderSource ) )
    {
        REX_DEBUG_LOG( "Failed to compile vertex shader. Log:\n", _program.GetVertexShader().GetLog() );
        return;
    }

    // attempt to compile the fragment shader
    if ( !_program.GetFragmentShader().LoadSource( fragmentShaderSource ) )
    {
        REX_DEBUG_LOG( "Failed to compile fragment shader. Log:\n", _program.GetFragmentShader().GetLog() );
        return;
    }

    // attempt to link the program
    if ( !_program.Link() )
    {
        REX_DEBUG_LOG( "Failed to link program. Log:\n", _program.GetLog() );
        return;
    }

    // now get the location of the tex variable
    _uniformTexLocation = _program.FindUniformLocation( "tex" );
}

// render the texture
void TextureRenderer::Render()
{
    // use the program and set the texture
    _program.Use();
    _program.SetTexture( _uniformTexLocation, *_texture, 0 );

    // draw the VAO
    glBindVertexArray( _vao );
    glDrawArrays( GL_TRIANGLES, 0, 6 );
    glBindVertexArray( 0 );
}

REX_NS_END