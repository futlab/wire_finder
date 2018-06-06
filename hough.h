#ifndef HOUGH_H
#define HOUGH_H

#include <map>

// OpenCV
#include <opencv2/core.hpp>

// OpenCL
#include "cl_utils.h"

void houghTest(CLSet *set);

class HoughLinesV
{
private:
    //cl::Context context_;
    //cl::CommandQueue queue_;
    //cl::Device device_;
    CLSet *set_;
    cl::Kernel kAccumulate_, kGrab;
    void loadKernels(const std::string &fileName, const std::vector<std::pair<std::string, int>> &params);
    cl::NDRange localSize_, scanGlobalSize_;
    cl::Buffer source_, accs_;
    cv::Size sourceSize_, accsSize_;
public:
    void initialize(const cv::Size &size, std::map<std::string, int> *paramsOut = nullptr);
    HoughLinesV(CLSet *set);
    void find(const cv::Mat &source, cv::Mat &result);
    void accumulate(const cv::Mat &source);
    void readAccumulator(cv::Mat &result);
};

#endif
