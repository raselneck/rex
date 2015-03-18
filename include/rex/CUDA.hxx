#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#if defined(__both__)
#  undef __both__
#endif

#define __cuda_func__ __host__ __device__
#define __cuda_const__ __constant__ const