# Rex the Ray Tracer

Rex is a ray tracer I am working on for an independent study where I will
be attempting to make a real-time ray tracer on the GPU utilizing CUDA. I
am using "Ray Tracing from the Ground Up" by Kevin Suffern as the basis and
reference for the ray tracer itself.

## Building & Running

To build and run the current version of Rex, you will need to check the
following:

1.   Make sure you have an NVIDIA GPU.
2.   Make sure your GPU supports CUDA 6.5+.
3.   Make sure your GPU supports OpenGL 4.0+.
4.   Have the CUDA SDK v6.5 installed.
     * A newer version of CUDA may work, but I use 6.5.
5.   Have Visual Studio 2013 installed.
     * This may work with an earlier version of VS, but I use 2013.
     * This may work on Linux and/or Mac, but I use Windows.

If you have all of the above steps complete, then you *should* be fine
to open up the solution file in Visual Studio and compile/run Rex.

If you don't have one or more of the above steps complete, then you're on
your own. Sorry. I don't have enough time to test this everywhere I can.

### What happened to the CPU version??

Not to fear! If you don't want to play around with CUDA, you can still
access the CPU-based version of Rex [here](https://github.com/fastinvsqrt/rex/tree/CPU)!

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