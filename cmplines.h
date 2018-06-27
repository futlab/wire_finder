#pragma once
// OpenCL
#include "cl_utils.h"

#include "defs.h"

class LinesCompare
{
private:
	cl::Set *set_;
	cl::Kernel kCompareLinesStereo_;
	cl::Counter cCompareLinesStereo_;
	cl::NDRange localSize_;
	cl::BufferT<uint> result_;
	cl::BufferT<LineV> leftLines_, rightLines_;
	void loadKernels(const std::string &fileName, const std::vector<std::pair<std::string, int> > &params);
public:
	void initialize(cl::MatBuffer &left, cl::MatBuffer &right);
	void stereoCompare(const std::vector<LineV> &left, const std::vector<LineV> &right, std::vector<uint> &result);
	LinesCompare(cl::Set *set);

};
