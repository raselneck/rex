# Rex the Ray Tracer

Rex is a ray tracer I am working on for an independent study where I will
be attempting to make a real-time ray tracer on the GPU utilizing either
CUDA or OpenCL (though, Rex is currently CPU-only). I am using "Ray Tracing
from the Ground Up" by Kevin Suffern as the basis and reference.

## Building

Rex is currently only set up to be used with Visual Studio 2013, but it
should be trivial to use other compilers, such as GCC or Clang.

All of the necessary include (`.h` and `.hxx`) files are in the `include`
directory, and all of the necessary source (`.cxx`) files are in the
`source` folder.

The following command works with G++ 4.8.2: `g++ ../source/*.cxx -I../include
-std=c++11 -w -O3 -o rex`

## License

```
The MIT License (MIT)

Copyright (c) 2015 Richard Selneck

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```