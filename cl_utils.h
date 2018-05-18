#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <string>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class CLWrapper
{
public:
    CLWrapper();
    std::vector<cl_kernel> loadKernels(const std::string sourceName, const std::vector<std::string> kernelNames);
    cl_mem createBuffer(size_t size);
    //cl_b
private:
    cl_context context_;
    cl_device_id deviceId_;
};

#endif // CL_UTILS_H
