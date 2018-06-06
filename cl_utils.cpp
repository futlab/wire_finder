#include "cl_utils.h"
#include <iostream>

CLSet::CLSet(cl_context context, cl_command_queue queue, std::vector<cl_device_id> devices) :
    context(context), queue(queue)
{
    for (auto &d : devices)
        this->devices.emplace_back(d);
}

size_t CLSet::getLocalSize()
{
    size_t result = 0;
    for (auto &device : devices) {
        size_t size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        if (!result || result > size) result = size;
    }
    return result;
}

void CLSet::initializeDefault()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform &platform = platforms[0];

    platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

    /* создать контекст */
    context = cl::Context(devices);// clCreateContext(NULL, 1, &deviceId_, NULL, NULL, &ret);

    /* создаем очередь команд */
    queue = cl::CommandQueue(context, devices[0]);// clCreateCommandQueue(context_, deviceId_, 0, &ret);
}

void printCLDevices()
{
    using namespace std;
    vector<cl::Platform> platforms;
    /* получить доступные платформы */
    cl::Platform::get(&platforms);
    //if(ret) throw  CLError(message, ret);
    //clCheck(ret, "Error in clGetPlatformIDs()");
    for(auto &platform : platforms) {
        cout << "Platform vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << endl;
        cout << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
        cout << "Platform profile: " << platform.getInfo<CL_PLATFORM_PROFILE>() << endl;
        cout << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() << endl;

        vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for(auto &device : devices) {
            cout << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << endl;
            cout << "Device profile: " << device.getInfo<CL_DEVICE_PROFILE>() << endl;
            cout << "Device image suport: " << device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() << endl;
        }
        cout << endl;
    }
}
