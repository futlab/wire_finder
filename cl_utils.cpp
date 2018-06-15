#include "cl_utils.h"
#include <iostream>
#include <fstream>

namespace cl
{

	Set::Set(cl_context context, cl_command_queue queue, std::vector<cl_device_id> devices) :
		context(context), queue(queue)
	{
		for (auto &d : devices)
			this->devices.emplace_back(d);
	}

	size_t Set::getLocalSize()
	{
		size_t result = 0;
		for (auto &device : devices) {
			size_t size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
			if (!result || result > size) result = size;
		}
		return result;
	}

	void Set::initializeDefault(const std::string &preferPlatform, const std::string &preferDevice)
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		cl::Platform platform = platforms[0];
		if (preferPlatform != "") 
			for (auto &p : platforms)
				if (p.getInfo<CL_PLATFORM_NAME>().find(preferPlatform) != std::string::npos) {
					platform = p;
					break;
				}
		std::cout << "Using platform " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

		platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

		/* создать контекст */
		context = cl::Context(devices);// clCreateContext(NULL, 1, &deviceId_, NULL, NULL, &ret);

		/* создаем очередь команд */
		queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);// clCreateCommandQueue(context_, deviceId_, 0, &ret);
	}

	Program Set::buildProgram(const std::string & fileName, const std::string & defines)
	{
		using namespace std;
		ifstream stream(fileName, ios_base::in);
		string source((istreambuf_iterator<char>(stream)),
			(std::istreambuf_iterator<char>()));

		cl::Program program(context, defines + source);
		try {
			program.build(devices);
		}
		catch (cl::BuildError &e) {
			for (auto &device : devices) {
				// Check the build status
				cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
				if (status != CL_BUILD_ERROR) continue;

				// Get the build log
				string name = device.getInfo<CL_DEVICE_NAME>();
				string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
				std::cerr << "Build log for " << fileName << " on device " << name << ":" << std::endl
					<< buildlog << std::endl;
			}
			throw;
		}
		std::cout << "Built " << fileName << " with defines:" << std::endl << defines;

		return program;
	}

	void printCLDevices()
	{
		using namespace std;
		vector<cl::Platform> platforms;
		/* получить доступные платформы */
		cl::Platform::get(&platforms);
		for (auto &platform : platforms) {
			cout << "Platform vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << endl;
			cout << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
			cout << "Platform profile: " << platform.getInfo<CL_PLATFORM_PROFILE>() << endl;
			cout << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() << endl;

			vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			for (auto &device : devices) {
				cout << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << endl;
				cout << "Device profile: " << device.getInfo<CL_DEVICE_PROFILE>() << endl;
				cout << "Device image suport: " << device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() << endl;
			}
			cout << endl;
		}
	}

	inline constexpr uint cvTypeSize(int type)
	{
		switch (type) {
		case CV_8U: return 1;
		case CV_16U: return 2;
		default: return 0;
		}
	}


	inline MatBuffer::MatBuffer(const MatBuffer & source) : 
		Buffer(source), size_(source.size()), set_(source.set_)
	{
	}

	MatBuffer::MatBuffer(Set * set, cv::Size size, int type, cl_mem_flags flags) :
		Buffer(set->context, flags, size.area() * cvTypeSize(type)), size_(size), type_(type), set_(set)
	{
	}

	MatBuffer & MatBuffer::operator=(const MatBuffer & buf)
	{
		Buffer::operator=(buf);
		size_ = buf.size_;
		type_ = buf.type_;
		set_ = buf.set_;
		return *this;
	}

	void MatBuffer::read(cv::Mat & result, bool blocking)
	{
		if (result.empty() || result.type() != type_ || result.size() != size_)
			result = cv::Mat(size_, type_);
		set_->queue.enqueueReadBuffer(*this, blocking, 0, size_.area() * cvTypeSize(type_), result.data);
	}

	cv::Mat MatBuffer::read()
	{
		cv::Mat result(size_, type_);
		set_->queue.enqueueReadBuffer(*this, true, 0, size_.area() * cvTypeSize(type_), result.data);
		return result;
	}

	cv::Mat MatBuffer::readScaled()
	{
		cv::Mat result(size_, type_);
		read(result);
		double min, max;
		cv::minMaxLoc(result, &min, &max);
		result.convertTo(result, CV_8U, 255 / max);
		return result;
	}

	void MatBuffer::write(const cv::Mat & source, bool blocking)
	{
		assert(!source.empty() && source.type() == type_ && source.size() == size_);
		set_->queue.enqueueWriteBuffer(*this, blocking, 0, size_.area() * cvTypeSize(type_), source.data);
	}

}