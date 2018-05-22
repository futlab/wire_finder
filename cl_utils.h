#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <string>
#include <vector>
#include <stdexcept>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class CLError : public std::runtime_error
{
public:
    CLError(const std::string &message, cl_int ret) : std::runtime_error(message + " Code: " + std::to_string(ret)) {}
};

class CLWrapper
{
public:
    CLWrapper();
    std::vector<cl_kernel> loadKernels(const std::string sourceName, const std::vector<std::string> kernelNames);
    cl_mem createBuffer(size_t size);
	void exec(cl_kernel kernel, const std::vector<size_t> &sizes);
	cl_int devInfo(cl_device_info param);
	std::string devInfoStr(cl_device_info param);
	template<class T> cl_int devInfo(cl_device_info param, T *value) { size_t size;  return clGetDeviceInfo(deviceId_, param, sizeof(*value), value, &size); }
	void finish();
	void printBuildInfo(cl_program program);
	void getImage2DFormats();
	//cl_b
private:
	friend class CLAbstractMem;
	friend class CLImage2D;
    cl_context context_;
    cl_device_id deviceId_;
	cl_command_queue commandQueue_;
};

class CLAbstractMem
{
public:
	CLAbstractMem(CLWrapper *cl, size_t size);
	void write(const void *data);
	void read(void *data);
	void setKernelArg(cl_kernel kernel, int arg);
protected:
	CLWrapper * const cl_;
	cl_mem memobj_;
	const size_t size_;
	inline cl_context context() { return cl_->context_; }
};

class CLMemory : public CLAbstractMem
{
public:
	CLMemory(CLWrapper *cl, size_t size);
};



template<class T = unsigned char, int SIZE = 1>
constexpr cl_image_format imageFormat();

template<> constexpr cl_image_format imageFormat<unsigned char, 1>() { return {CL_R, CL_UNSIGNED_INT8 }; }
template<> constexpr cl_image_format imageFormat<signed char, 1>() { return { CL_R, CL_SIGNED_INT8 }; }
template<> constexpr cl_image_format imageFormat<unsigned char, 4>() { return { CL_RGBA, CL_UNSIGNED_INT8 }; }
template<> constexpr cl_image_format imageFormat<signed char, 4>() { return { CL_RGBA, CL_SIGNED_INT8 }; }
template<> constexpr cl_image_format imageFormat<cl_ushort, 4>() { return { CL_RGBA, CL_UNSIGNED_INT16 }; }
template<> constexpr cl_image_format imageFormat<cl_short, 4>() { return { CL_RGBA, CL_SIGNED_INT16 }; }
template<> constexpr cl_image_format imageFormat<cl_ushort, 2>() { return { CL_RG, CL_UNSIGNED_INT16 }; }
template<> constexpr cl_image_format imageFormat<cl_short, 2>() { return { CL_RG, CL_SIGNED_INT16 }; }



class CLImage2D : public CLAbstractMem
{
public:
    CLImage2D(CLWrapper *cl, size_t width, size_t height, void *data = nullptr, const cl_image_format &format = imageFormat(), cl_mem_flags flags = CL_MEM_READ_WRITE);
	void read(void *data);
	void write(void *data);
private:
    size_t w, h;
};

#endif // CL_UTILS_H
