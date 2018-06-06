#ifndef CL_UTILS_H
#define CL_UTILS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include "cl2.hpp"
#include <vector>

struct CLSet
{
   cl::Context context;
   cl::CommandQueue queue;
   std::vector<cl::Device> devices;
   CLSet() {}
   CLSet(cl_context context, cl_command_queue queue, std::vector<cl_device_id> devices);
   size_t getLocalSize();
};

void printCLDevices();

#endif // CL_UTILS_H
