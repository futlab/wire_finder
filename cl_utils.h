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
	void exec(cl_kernel kernel, const std::vector<size_t> &sizes);
	cl_int devInfo(cl_device_info param);
	std::string devInfoStr(cl_device_info param);
	template<class T> cl_int devInfo(cl_device_info param, T *value) { size_t size;  return clGetDeviceInfo(deviceId_, param, sizeof(*value), value, &size); }
	void finish();
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



//template<class T, int SIZE>
//constexpr cl_image_format getImageFormat();

//template<> cl_image_format getImageFormat<

class CLImage2D : public CLAbstractMem
{
public:
	CLImage2D(CLWrapper *cl, int width, int height, void *data = nullptr, cl_mem_flags flags = CL_MEM_READ_WRITE);
	void read(void *data);
	void write(void *data);
private:
	int w, h;
};

#endif // CL_UTILS_H
