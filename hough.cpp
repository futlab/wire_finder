#include "hough.h"
#include <fstream>
#include <iostream>

using namespace std;

void HoughLinesV::loadKernels(const string &fileName, const vector<pair<string, int> > &params)
{
    using namespace cl;

    ifstream stream(fileName, ios_base::in);
    string source( (istreambuf_iterator<char>(stream) ),
                           (std::istreambuf_iterator<char>()    ) );
    string defines = "";
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
    kGrab = cl::Kernel(program, "glueAccs");

    size_t groupSize = kAccumulate_.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(set_->devices[0]);
    localSize_ = NDRange(groupSize);
}

void HoughLinesV::initialize(const cv::Size &size, std::map<string, int> *paramsOut)
{
    size_t wy = 45, accH = 128;
    size_t localMemorySize = set_->getLocalSize();
    size_t wx = localMemorySize / accH - wy + 1;
    size_t horGroups = (size.width + wx - 1) / wx;
    size_t verGroups = size.height / wy;
    wx = size.width / horGroups;
    const vector<pair<string, int> > params = {
        {"WX",      wx},
        {"WY",      wy},
        {"ACC_H", accH},
        {"WIDTH", size.width}
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
    accs_ = cl::Buffer(set_->context, CL_MEM_READ_WRITE, accsSize_.area());

    // Initialize kernel parameters
    kAccumulate_.setArg(0, source_);
    cl_uint step = size.width;
    kAccumulate_.setArg(1, step);
    kAccumulate_.setArg(2, accs_);

    kGrab.setArg(0, accs_);
    //kGrab.setArg(1);
}

HoughLinesV::HoughLinesV(CLSet *set) :
    set_(set)
{
}

void HoughLinesV::find(const cv::Mat &source, cv::Mat &result)
{
    using namespace cl;
    accumulate(source);
    set_->queue.enqueueNDRangeKernel(kGrab, NDRange(), NDRange(localSize_[0] * 128), localSize_);
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
        result = cv::Mat(accsSize_, CV_8U);
    set_->queue.enqueueReadBuffer(accs_, true, 0, accsSize_.area(), result.data);
}

