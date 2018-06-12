#include "hough.h"
#include <fstream>
#include <iostream>

using namespace std;

std::string cvTypeToCL(int type)
{
	switch (type) {
	case CV_8U: return "uchar";
	case CV_16U: return "ushort";
	default: return "???";
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

void HoughLinesV::loadKernels(const string &fileName, const vector<pair<string, int> > &params)
{
    using namespace cl;

    ifstream stream(fileName, ios_base::in);
    string source( (istreambuf_iterator<char>(stream) ),
                           (std::istreambuf_iterator<char>()    ) );
	string defines = "#define ACC_TYPE " + cvTypeToCL(accType_) + "\n";
    for (const auto &p : params)
        defines += "#define " + p.first + " " + std::to_string(p.second) + "\n";
    source = defines + source;

    cl::Program program(set_->context, source);
    try {
        program.build(set_->devices);
    } catch(cl::BuildError &e) {
        for (auto &device : set_->devices) {
            // Check the build status
            cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
            if (status != CL_BUILD_ERROR) continue;

            // Get the build log
            string name     = device.getInfo<CL_DEVICE_NAME>();
            string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            cerr << "Build log for " << fileName << " on device " << name << ":" << endl
                    << buildlog << endl;
        }
    }
    kAccumulate_ = cl::Kernel(program, "accumulate");
    kCollectLines_ = cl::Kernel(program, "collectLines");

    size_t groupSize = kAccumulate_.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(set_->devices[0]);
    localSize_ = NDRange(groupSize);
}

void HoughLinesV::initialize(const cv::Size &size, int accType, std::map<string, int> *paramsOut)
{
	accType_ = accType;
    size_t wy = 45, accH = 128, maxLines = 1024;
    size_t localMemorySize = set_->getLocalSize();
    size_t wx = localMemorySize / (accH * cvTypeSize(accType)) - wy + 1;
    size_t horGroups = (size.width + wx - 1) / wx;
    size_t verGroups = size.height / wy;
    wx = size.width / horGroups;
    const vector<pair<string, int> > params = {
        {"WX", (int)wx},
        {"WY", (int)wy},
        {"ACC_H", accH},
        {"WIDTH", size.width},
		{"MAX_LINES", maxLines}
    };
    loadKernels("hough.cl", params);
    if (paramsOut) {
        paramsOut->clear();
        for (const auto &p : params)
            paramsOut->insert(p);
    }
    scanGlobalSize_ = cl::NDRange(horGroups * localSize_[0], verGroups);

    source_ = cl::Buffer(set_->context, CL_MEM_READ_ONLY, size.area());
    sourceSize_ = size;
    accsSize_ = cv::Size(wx + wy - 1, accH * verGroups * horGroups);
    accs_ = cl::Buffer(set_->context, CL_MEM_READ_WRITE, accsSize_.area() * cvTypeSize(accType_));
	linesCount_ = cl::Buffer(set_->context, CL_MEM_WRITE_ONLY, sizeof(uint));
	lines_ = cl::Buffer(set_->context, CL_MEM_READ_WRITE, maxLines * sizeof(LineV));

    // Initialize kernel parameters
    kAccumulate_.setArg(0, source_);
    cl_uint step = size.width;
    kAccumulate_.setArg(1, step);
    kAccumulate_.setArg(2, accs_);

    //kGrab.setArg(0, accs_);
    //kGrab.setArg(1);

	kCollectLines_.setArg(0, acc_);
	uint collectThreshold = 10;
	kCollectLines_.setArg(1, collectThreshold);
	//kCollectLines_.setArg(2, step);
	kCollectLines_.setArg(3, linesCount_);
	kCollectLines_.setArg(4, lines_);
}

HoughLinesV::HoughLinesV(CLSet *set) :
    set_(set)
{
}

void HoughLinesV::find(const cv::Mat &source, cv::Mat &result)
{
    using namespace cl;
    accumulate(source);
    //set_->queue.enqueueNDRangeKernel(kGrab, NDRange(), NDRange(localSize_[0] * 128), localSize_);
}

void HoughLinesV::accumulate(const cv::Mat &source)
{
    using namespace cl;
    set_->queue.enqueueWriteBuffer(source_, true, 0, sourceSize_.area(), source.data);
    set_->queue.enqueueNDRangeKernel(kAccumulate_, NDRange(), scanGlobalSize_, localSize_);
}

void HoughLinesV::readAccumulator(cv::Mat &result)
{
    if (result.empty())
        result = cv::Mat(accsSize_, accType_);
    set_->queue.enqueueReadBuffer(accs_, true, 0, accsSize_.area() * cvTypeSize(accType_), result.data);
}

void HoughLinesV::readLines(std::vector<LineV>& lines)
{
	uint size;
	set_->queue.enqueueReadBuffer(linesCount_, true, 0, sizeof(uint), &size);
	lines.resize(size);
	if (size)
		set_->queue.enqueueReadBuffer(lines_, true, 0, size * sizeof(LineV), lines.data());
}

void HoughLinesV::collectLines()
{
	//uint width;
	//set_->queue.enqueueNDRangeKernel(kCollectLines_, cl::NDRange(), cl::NDRange(width), localSize_);
}

void HoughLinesV::collectLines(const cv::Mat & acc)
{
	using namespace cl;
	acc_ = Buffer(set_->context, CL_MEM_READ_WRITE, acc.total());
	kCollectLines_.setArg(0, acc_);
	uint width = acc.cols;
	kCollectLines_.setArg(2, width);

	NDRange globalSize(width / localSize_[0] * localSize_[0]);
	set_->queue.enqueueWriteBuffer(acc_, false, 0, acc.total(), acc.data);
	set_->queue.enqueueNDRangeKernel(kCollectLines_, NDRange(), globalSize, localSize_);
}

