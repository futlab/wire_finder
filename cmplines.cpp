#include "cmplines.h"

using namespace std;

LinesCompare::LinesCompare(cl::Set * set) : set_(set),
cCompareLines_("compareLines")
{
	/*bytesAlign_ = 0;
	for (auto &d : set->devices) {
	uint ba = d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
	if (ba > bytesAlign_) bytesAlign_ = ba;
	}*/
}

void LinesCompare::loadKernels(const string &fileName, const vector<pair<string, int> > &params)
{
	using namespace cl;

	string defines = "";
	for (const auto &p : params)
		defines += "#define " + p.first + " " + std::to_string(p.second) + "\n";
	Program program = set_->buildProgram(fileName, defines);

	kCompareLines_ = Kernel(program, "compareLines");

	size_t groupSize = kCompareLines_.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(set_->devices[0]);
	localSize_ = NDRange(groupSize);
}

void LinesCompare::initialize(cl::MatBuffer & left, cl::MatBuffer & right)
{
	using namespace cl;
	assert(left.size() == right.size() && left.type() == right.type());
	cv::Size size = left.size();
	loadKernels("cmplines.cl", {
		{ "WIDTH", size.width },
		{ "HEIGHT", size.height },
	});

	leftLines_	= BufferT<LineV>(set_, 64);
	rightLines_ = BufferT<LineV>(set_, 64);
	result_		= BufferT<uint>(set_, 32 * 32);

	kCompareLines_.setArg(0, left);
	kCompareLines_.setArg(1, right);
	kCompareLines_.setArg(2, leftLines_);
	kCompareLines_.setArg(3, rightLines_);
	kCompareLines_.setArg(5, result_);
}

void LinesCompare::compare(const vector<LineV> &left, const vector<LineV> &right, vector<uint> &result)
{
	size_t leftCount = left.size(), rightCount = right.size();
	if (!leftCount || !rightCount)
		return;
	if (leftCount * rightCount > 32 * 32) {
		if (leftCount > 32) leftCount = 32;
		if (rightCount > 32) rightCount = 32;
	}
	if (leftCount > 64) leftCount = 64;
	if (rightCount > 64) rightCount = 64;

	kCompareLines_.setArg(4, uint(rightCount));

	leftLines_.write(left, leftCount);
	rightLines_.write(right, rightCount);
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kCompareLines_, cl::NDRange(), cl::NDRange(localSize_[0] * leftCount), localSize_, nullptr, &e);
	cCompareLines_.inc(e);
	result_.read(result, leftCount * rightCount);
}

