#include "cl_utils.h"

#include <fstream>
#include <algorithm>

CLWrapper::CLWrapper()
{
    int ret;
    cl_platform_id platformId;
    cl_uint numPlatforms;
    /* получить доступные платформы */
    ret = clGetPlatformIDs(1, &platformId, &numPlatforms);

    cl_uint numDevices;
    /* получить доступные устройства */
    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId_, &numDevices);

    /* создать контекст */
    context_ = clCreateContext(NULL, 1, &deviceId_, NULL, NULL, &ret);

    /* создаем команду */
    cl_command_queue command_queue = clCreateCommandQueue(context_, deviceId_, 0, &ret);
}

std::vector<cl_kernel> CLWrapper::loadKernels(const std::string sourceName, const std::vector<std::string> kernelNames)
{
    std::ifstream stream(sourceName, std::ios_base::in);
    std::string source( (std::istreambuf_iterator<char>(stream) ),
                           (std::istreambuf_iterator<char>()    ) );

    const char *sourcePtr = source.c_str();
    size_t size = source.size();
    cl_int ret;
    cl_program program = clCreateProgramWithSource(context_, 1, &sourcePtr, &size, &ret);

    ret = clBuildProgram(program, 1, &deviceId_, NULL, NULL, NULL);

    std::vector<cl_kernel> result;
    for (const auto &name : kernelNames) {
        cl_kernel kernel = clCreateKernel(program, name.c_str(), &ret);
        result.push_back(kernel);
    }
    return result;
}

cl_mem CLWrapper::createBuffer(size_t size)
{
    cl_int ret;
    auto memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
    return memobj;
}
