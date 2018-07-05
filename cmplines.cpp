#include "cmplines.h"

using namespace std;

LinesCompare::LinesCompare(cl::Set * set) : set_(set),
cCompareLinesStereo_("compareStereo"), cCompareLinesAdjacent_("compareAdjacent")
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

	kCompareLinesStereo_ = Kernel(program, "compareLinesStereo");
	kCompareLinesAdjacent_ = Kernel(program, "compareLinesAdjacent");

	size_t groupSize = kCompareLinesStereo_.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(set_->devices[0]);
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

	kCompareLinesStereo_.setArg(0, left);
	kCompareLinesStereo_.setArg(1, right);
	kCompareLinesStereo_.setArg(2, leftLines_);
	kCompareLinesStereo_.setArg(3, rightLines_);
	kCompareLinesStereo_.setArg(5, result_);

	kCompareLinesAdjacent_.setArg(0, left);
	kCompareLinesAdjacent_.setArg(1, right);
	kCompareLinesAdjacent_.setArg(2, leftLines_);
	kCompareLinesAdjacent_.setArg(3, rightLines_);
	kCompareLinesAdjacent_.setArg(6, result_);
}

void LinesCompare::stereoCompare(const vector<LineV> &left, const vector<LineV> &right, vector<uint> &result)
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

	kCompareLinesStereo_.setArg(4, uint(rightCount));

	leftLines_.write(left, leftCount);
	rightLines_.write(right, rightCount);
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kCompareLinesStereo_, cl::NDRange(), cl::NDRange(localSize_[0] * leftCount), localSize_, nullptr, &e);
	cCompareLinesStereo_.inc(e);
	result_.read(result, leftCount * rightCount);
}

void LinesCompare::adjacentCompare(const std::vector<LineV>& first, const std::vector<LineV>& second, float twist, std::vector<uint>& result)
{
	size_t firstCount = first.size(), secondCount = second.size();
	if (!firstCount || !secondCount)
		return;
	if (firstCount * secondCount > 32 * 32) {
		if (firstCount > 32) firstCount = 32;
		if (secondCount > 32) secondCount = 32;
	}
	if (firstCount > 64) firstCount = 64;
	if (secondCount > 64) secondCount = 64;

	kCompareLinesAdjacent_.setArg(4, uint(secondCount));
	cl_int twistInt = (cl_int)(tan(twist) * 32768);
	kCompareLinesAdjacent_.setArg(5, twistInt);

	leftLines_.write(first, firstCount);
	rightLines_.write(second, secondCount);
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kCompareLinesAdjacent_, cl::NDRange(), cl::NDRange(localSize_[0] * firstCount), localSize_, nullptr, &e);
	cCompareLinesStereo_.inc(e);
	result_.read(result, firstCount * secondCount);
}

void LinesCompare::adjacentCompare(cl::MatBuffer & firstImage, cl::MatBuffer & secondImage, const std::vector<LineV>& first, const std::vector<LineV>& second, float twist, std::vector<uint>& result)
{
	kCompareLinesAdjacent_.setArg(0, firstImage);
	kCompareLinesAdjacent_.setArg(1, secondImage);
	adjacentCompare(first, second, twist, result);
}

