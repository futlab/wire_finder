#include "cl_utils.h"

#include <fstream>
#include <algorithm>

inline void clCheck(cl_int ret, const std::string &message)
{
    if(ret) throw CLError(message, ret);
}

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
	if (ret) {
		printBuildInfo(program);
		return {};
	}

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
    if (ret) throw CLError("Cannot createBuffer()", ret);
    return memobj;
}

void CLWrapper::exec(cl_kernel kernel, const std::vector<size_t> &sizes)
{
	cl_int ret = clEnqueueNDRangeKernel(commandQueue_, kernel, (cl_uint)sizes.size(), NULL, sizes.data(), NULL, 0, NULL, NULL);
    if (ret) throw CLError("Cannot clEnqueueNDRangeKernel()", ret);
}

cl_int CLWrapper::devInfo(cl_device_info param)
{
	cl_int value, ret;
	size_t size;
	ret = clGetDeviceInfo(deviceId_, param, sizeof(value), &value, &size);
    if(ret) throw CLError("Cannot clGetDeviceInfo()", ret);
	return value;
}

std::string CLWrapper::devInfoStr(cl_device_info param)
{
	char buf[64] = {};
	size_t size;
	cl_int ret = clGetDeviceInfo(deviceId_, param, sizeof(buf) - 1, buf, &size);
    if(ret) throw CLError("Cannot clGetDeviceInfo()", ret);
    return buf;
}

void CLWrapper::finish()
{
	clFinish(commandQueue_);
}

inline void CLWrapper::printBuildInfo(cl_program program)
{
	char result[8192];
	size_t size;
	clGetProgramBuildInfo(program, deviceId_, CL_PROGRAM_BUILD_LOG, sizeof(result), result, &size);
	printf("%s\n", result);
}

void CLWrapper::getImage2DFormats()
{
	cl_image_format formats[512];
	cl_uint count;
	clGetSupportedImageFormats(context_, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 512, formats, &count);
	for (cl_image_format *f = formats; count; --count, ++f) {
		std::string text;
		switch (f->image_channel_data_type) {
		case CL_UNSIGNED_INT16: text += "UNSIGNED_INT16"; break;
		case CL_SIGNED_INT16: text += "SIGNED_INT16"; break;
		case CL_SNORM_INT8: text += "SNORM_INT8"; break;
		case CL_SNORM_INT16: text += "SNORM_INT16"; break;
		case CL_UNORM_INT8: text += "UNORM_INT8"; break;
		case CL_UNORM_INT16: text += "UNORM_INT16"; break;
		case CL_SIGNED_INT8: text += "SIGNED_INT8"; break;
		case CL_UNSIGNED_INT8: text += "UNSIGNED_INT8"; break;
		case CL_SIGNED_INT32: text += "SIGNED_INT32"; break;
		case CL_UNSIGNED_INT32: text += "UNSIGNED_INT32"; break;
		case CL_FLOAT: text += "FLOAT"; break;
		case CL_HALF_FLOAT: text += "HALF_FLOAT"; break;
		default: text += "???"; break;
		}
		text += " ";
		switch (f->image_channel_order) {
		case CL_R: text += "R"; break;
		case CL_A: text += "A"; break;
		case CL_RG: text += "RG"; break;
		case CL_RGB: text += "RGB"; break;
		case CL_ARGB: text += "ARGB"; break;
		case CL_RGBA: text += "RGBA"; break;
		case CL_BGRA: text += "BGRA"; break;
		case CL_RA: text += "RA"; break;
		case CL_DEPTH: text += "DEPTH"; break;
		case CL_DEPTH_STENCIL: text += "DEPTH_STENCIL"; break;
		case CL_INTENSITY: text += "INTENSITY"; break;
		case CL_LUMINANCE: text += "LUMINANCE"; break;
		default: text += "???"; break;
		}
        printf("%s\n", text.c_str());
	}
}

CLAbstractMem::CLAbstractMem(CLWrapper * cl, size_t size) : cl_(cl), size_(size)
{
}

void CLAbstractMem::write(const void * data)
{
	cl_int ret;
	ret = clEnqueueWriteBuffer(cl_->commandQueue_, memobj_, CL_TRUE, 0, size_, data, 0, NULL, NULL);
    if(ret) throw CLError("Cannot clEnqueueWriteBuffer()", ret);
}

void CLAbstractMem::read(void * data)
{
	cl_int ret = clEnqueueReadBuffer(cl_->commandQueue_, memobj_, CL_TRUE, 0, size_, data, 0, NULL, NULL);
    if(ret) throw CLError("Cannot clEnqueueReadBuffer()", ret);
}

void CLAbstractMem::setKernelArg(cl_kernel kernel, int arg)
{
	cl_int ret = clSetKernelArg(kernel, arg, sizeof(cl_mem), (void *)&memobj_);
    if(ret)
        throw CLError("Cannot clSetKernelArg()", ret);
}

CLMemory::CLMemory(CLWrapper * cl, size_t size) : CLAbstractMem(cl, size)
{
	cl_int ret;
	memobj_ = clCreateBuffer(context(), CL_MEM_READ_WRITE, size, NULL, &ret);
    if(ret) throw CLError("Cannot clCreateBuffer()", ret);
}

CLImage2D::CLImage2D(CLWrapper * cl, size_t width, size_t height, void * data, const cl_image_format &format, cl_mem_flags flags) :
    CLAbstractMem(cl, width * height), w(width), h(height)
{
	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = width;
	desc.image_height = height;
	cl_int ret;
	memobj_ = clCreateImage(context(), flags, &format, &desc, data, &ret);
    if(ret) throw CLError("Cannot clCreateImage()", ret);
}

void CLImage2D::read(void * data)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { w, h, 1 };
	cl_int ret = clEnqueueReadImage(cl_->commandQueue_, memobj_, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
    if(ret) throw CLError("clEnqueueReadImage()", ret);
}

void CLImage2D::write(void * data)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { w, h, 1 };
	cl_int ret = clEnqueueWriteImage(cl_->commandQueue_, memobj_, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
    if(ret) throw CLError("clEnqueueWriteImage()", ret);
}
