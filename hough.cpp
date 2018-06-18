#include "hough.h"

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

HoughLinesV::HoughLinesV(cl::Set *set) : set_(set),
	cAccumulate_("accumulate"), cAccumulateRows_("accumulateRows"), cSumAccumulator_("sumAccumulator"), cCollectLines_("collectLines")
{
	bytesAlign_ = 0;
	for (auto &d : set->devices) {
		uint ba = d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
		if (ba > bytesAlign_) bytesAlign_ = ba;
	}
	// if (bytesAlign_ > 64) bytesAlign_ = 64;
}


void HoughLinesV::loadKernels(const string &fileName, const vector<pair<string, int> > &params)
{
    using namespace cl;

	string defines = "#define ACC_TYPE " + cvTypeToCL(accType_) + "\n";
    for (const auto &p : params)
        defines += "#define " + p.first + " " + std::to_string(p.second) + "\n";
	Program program = set_->buildProgram(fileName, defines);
 
	kAccumulate_ = Kernel(program, "accumulate");
	kAccumulateRows_ = Kernel(program, "accumulateRows");
	kCollectLines_ = Kernel(program, "collectLines");
	kSumAccumulator_ = Kernel(program, "sumAccumulator");

    size_t groupSize = kAccumulate_.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(set_->devices[0]);
    localSize_ = NDRange(groupSize);
}

uint HoughLinesV::alignSize(uint size)
{
	if (bytesAlign_)
		return ((size - 1) | (bytesAlign_ / cvTypeSize(accType_) - 1)) + 1;
	return size;
}

std::string HoughLinesV::getCounters()
{
	return cAccumulate_.timeStr() + cAccumulateRows_.timeStr() + cSumAccumulator_.timeStr() + cCollectLines_.timeStr();
}

void HoughLinesV::initialize(const cv::Size &size, int accType, std::map<string, int> *paramsOut)
{
	accType_ = accType;
    uint wx = 160, wy = 45, accH = 128, maxLines = 1024;
    uint localMemorySize = (uint)set_->getLocalSize();
    //uint wx = localMemorySize / (accH * cvTypeSize(accType)) - wy + 1;
	uint horGroups = size.width / wx;// (size.width + wx - 1) / wx;
	uint verGroups = size.height / wy;
    //wx = size.width / horGroups;

	int accW = (int)alignSize(wx + wy - 1);
	assert(accW * accH * cvTypeSize(accType_) < localMemorySize);

	const vector<pair<string, int> > params = {
        {"WX", (int)wx},
        {"WY", (int)wy},
		{"ACC_W", accW},
        {"ACC_H", accH},
        {"WIDTH", size.width},
		{"HEIGHT", size.height},
		{"MAX_LINES", maxLines}
    };
    loadKernels("hough.cl", params);
    if (paramsOut) {
        paramsOut->clear();
        for (const auto &p : params)
            paramsOut->insert(p);
    }
	using namespace cl;
	scanGlobalSize_ = NDRange(horGroups * localSize_[0], verGroups);

    source_		= MatBuffer(set_, size, CV_8U, CL_MEM_READ_ONLY);
    accs_		= MatBuffer(set_, cv::Size(accW, accH * verGroups * horGroups), accType_);
	accRows_	= MatBuffer(set_, cv::Size(size.width + (accW - wx), accH * verGroups), accType_);
	linesCount_ = BufferT<uint>(set_);
	lines_		= BufferT<LineV>(set_, maxLines);
	flags_		= BufferT<ushort>(set_, scanGlobalSize_[0] * scanGlobalSize_[1] / localSize_[0]);

	auto &sourceSize = source_.size();
	uint accWidth = alignSize(sourceSize.width + sourceSize.height - 1);
	accumulator        = MatBuffer(set_, cv::Size(accWidth, accH), accType_);

    // Initialize kernel parameters
    
	// Kernel accumulate: source_, step => accs_
	kAccumulate_.setArg(0, source_);
    cl_uint step = size.width;		
    kAccumulate_.setArg(1, step);
    kAccumulate_.setArg(2, accs_);

	// Kernel accumulateRows: source_, step, flags_ => accRows_
	kAccumulateRows_.setArg(0, source_);
	kAccumulateRows_.setArg(1, step);
	kAccumulateRows_.setArg(2, accRows_);
	kAccumulateRows_.setArg(3, flags_);

	// Kernel sumAccumulator: accRows_ => acc_;
	kSumAccumulator_.setArg(0, accRows_);
	kSumAccumulator_.setArg(1, accumulator);

	// Kernel collectLines: acc_, threshold, step => linescount_, lines_
	kCollectLines_.setArg(0, accumulator);
	uint collectThreshold = 10;
	kCollectLines_.setArg(1, collectThreshold);
	uint accStep = accumulator.size().width;
	kCollectLines_.setArg(2, accStep);
	kCollectLines_.setArg(3, linesCount_);
	kCollectLines_.setArg(4, lines_);
}

void HoughLinesV::find(const cv::Mat &source, cv::Mat &result)
{
    using namespace cl;
    accumulate(source);
    //set_->queue.enqueueNDRangeKernel(kGrab, NDRange(), NDRange(localSize_[0] * 128), localSize_);
}

void HoughLinesV::accumulate(const cv::Mat &source)
{
	source_.write(source);
	cl::Event e;
    set_->queue.enqueueNDRangeKernel(kAccumulate_, cl::NDRange(), scanGlobalSize_, localSize_, nullptr, &e);
	cAccumulate_.inc(e);
}

void HoughLinesV::accumulateRows(const cv::Mat & source)
{
	source_.write(source);
	flags_.fill();
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kAccumulateRows_, cl::NDRange(), scanGlobalSize_, localSize_, nullptr, &e);
	cAccumulateRows_.inc(e);
}

void HoughLinesV::accumulateRows(const cv::Mat & source, cv::Mat & rows)
{
	accumulateRows(source);
	accRows_.read(rows);
}

void HoughLinesV::sumAccumulator()
{
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kSumAccumulator_, cl::NDRange(), cl::NDRange(localSize_[0] * 16), localSize_, nullptr, &e);
	cSumAccumulator_.inc(e);
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
	using namespace cl;
	NDRange globalSize(accumulator.size().width / localSize_[0] * localSize_[0]);
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kCollectLines_, NDRange(), globalSize, localSize_, nullptr, &e);
	cCollectLines_.inc(e);
}

void HoughLinesV::collectLines(const cv::Mat & source)
{
	accumulator.write(source);
	collectLines();
}

