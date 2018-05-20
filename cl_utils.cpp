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

    /* создаем очередь команд */
    commandQueue_ = clCreateCommandQueue(context_, deviceId_, 0, &ret);
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
    cl_mem memobj = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, NULL, &ret);
    return memobj;
}

void CLWrapper::exec(cl_kernel kernel, const std::vector<size_t> &sizes)
{
	cl_int ret = clEnqueueNDRangeKernel(commandQueue_, kernel, (cl_uint)sizes.size(), NULL, sizes.data(), NULL, 0, NULL, NULL);
}

cl_int CLWrapper::devInfo(cl_device_info param)
{
	cl_int value, ret;
	size_t size;
	ret = clGetDeviceInfo(deviceId_, param, sizeof(value), &value, &size);
	return value;
}

std::string CLWrapper::devInfoStr(cl_device_info param)
{
	char buf[64] = {};
	size_t size;
	cl_int ret = clGetDeviceInfo(deviceId_, param, sizeof(buf) - 1, buf, &size);
	return buf;
}

void CLWrapper::finish()
{
	clFinish(commandQueue_);
}

CLAbstractMem::CLAbstractMem(CLWrapper * cl, size_t size) : cl_(cl), size_(size)
{
}

void CLAbstractMem::write(const void * data)
{
	cl_int ret;
	ret = clEnqueueWriteBuffer(cl_->commandQueue_, memobj_, CL_TRUE, 0, size_, data, 0, NULL, NULL);
	_ASSERT(!ret);
}

void CLAbstractMem::read(void * data)
{
	cl_int ret = clEnqueueReadBuffer(cl_->commandQueue_, memobj_, CL_TRUE, 0, size_, data, 0, NULL, NULL);
	_ASSERT(!ret);
}

void CLAbstractMem::setKernelArg(cl_kernel kernel, int arg)
{
	cl_int ret = clSetKernelArg(kernel, arg, sizeof(cl_mem), (void *)&memobj_);
	_ASSERT(!ret);
}

CLMemory::CLMemory(CLWrapper * cl, size_t size) : CLAbstractMem(cl, size)
{
	cl_int ret;
	memobj_ = clCreateBuffer(context(), CL_MEM_READ_WRITE, size, NULL, &ret);
	_ASSERT(!ret);
}

CLImage2D::CLImage2D(CLWrapper * cl, int width, int height, void * data, cl_mem_flags flags) : CLAbstractMem(cl, width * height), w(width), h(height)
{
	cl_image_format format;
	format.image_channel_data_type = CL_UNSIGNED_INT8;
	format.image_channel_order = CL_R;
	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = width;
	desc.image_height = height;
	cl_int ret;
	memobj_ = clCreateImage(context(), flags, &format, &desc, data, &ret);
	//memobj_ = clCreateBuffer(context(), flags, size_, NULL, &ret);
	_ASSERT(!ret);
}

void CLImage2D::read(void * data)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { w, h, 1 };
	cl_int ret = clEnqueueReadImage(cl_->commandQueue_, memobj_, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
	_ASSERT(!ret);
}

void CLImage2D::write(void * data)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { w, h, 1 };
	cl_int ret = clEnqueueWriteImage(cl_->commandQueue_, memobj_, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
	_ASSERT(!ret);
}

