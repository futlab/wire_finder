#ifndef CL_UTILS_H
#define CL_UTILS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include "cl2.hpp"
#include <vector>
#include <opencv2/core.hpp>

namespace cl
{

	struct Set
	{
	   cl::Context context;
	   cl::CommandQueue queue;
	   std::vector<cl::Device> devices;
	   Set() {}
	   Set(cl_context context, cl_command_queue queue, std::vector<cl_device_id> devices);
	   size_t getLocalSize();
	   void initializeDefault();
	};


	class MatBuffer : public Buffer
	{
	private:
		cv::Size size_;
		int type_;
		Set *set_;
	public:
		inline const cv::Size &size() const { return size_; }
		inline const int type() const { return type_; }
		MatBuffer() {}
		MatBuffer(const MatBuffer &source);
		MatBuffer(Set *set, cv::Size size, int type = CV_8U, cl_mem_flags flags = CL_MEM_READ_WRITE);
		MatBuffer& operator = (const MatBuffer &buf);
		MatBuffer(MatBuffer&& buf) noexcept : Buffer(std::move(buf)), size_(buf.size_), type_(buf.type_), set_(buf.set_) {}
		void read(cv::Mat &result, bool blocking = true);
		void write(const cv::Mat &source, bool blocking = false);
	};

	template<typename T>
	class BufferT : public Buffer
	{
	private:
		size_t size_;
		Set *set_;
	public:
		inline const cv::Size &size() const { return size_; }
		inline void fill(T value = 0) { set_->queue.enqueueFillBuffer(*this, value, 0, size_); }

		BufferT<T>& operator = (const BufferT<T> &buf)
		{
			Buffer::operator=(buf);
			size_ = buf.size_;
			set_ = buf.set_;
			return *this;
		}

		BufferT(size_t count = 1) : size_(count * sizeof(T)) {}
		BufferT(Set *set, size_t count = 1, cl_mem_flags flags = CL_MEM_READ_WRITE) :
			Buffer(set->context, flags, count * sizeof(T)), size_(count * sizeof(T)), set_(set)
		{}
		BufferT(BufferT<T>&& buf) noexcept : Buffer(std::move(buf)), size_(buf.size_) {}
	};

	void printCLDevices();
}

#endif // CL_UTILS_H
