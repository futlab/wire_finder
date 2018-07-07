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
	cAccumulate_("accumulate"), cAccumulateRows_("accumulateRows"), cSumAccumulator_("sumAccumulator"), cCollectLines_("collectLines"), cRefineLines_("refineLines")
{
	bytesAlign_ = 0;
	for (auto &d : set->devices) {
		uint ba = d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
		if (ba > bytesAlign_) bytesAlign_ = ba;
	}
}


void HoughLinesV::loadKernels(const string &fileName, const vector<pair<string, int> > &params)
{
    using namespace cl;

	string defines = 
		"#define ACC_TYPE " + cvTypeToCL(accType_) + "\n" +
		"#define ROW_TYPE " + cvTypeToCL(rowType_) + "\n";
    for (const auto &p : params)
        defines += "#define " + p.first + " " + std::to_string(p.second) + "\n";
	Program program = set_->buildProgram(fileName, defines);
 
	kAccumulate_ = Kernel(program, "accumulate");
	kAccumulateRows_ = Kernel(program, "accumulateRows");
	kCollectLines_ = Kernel(program, "collectLines");
	kSumAccumulator_ = Kernel(program, "sumAccumulator");
	kRefineLines_ = Kernel(program, "refineLines");

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
	return cAccumulate_.timeStr() + cAccumulateRows_.timeStr() + cSumAccumulator_.timeStr() + cCollectLines_.timeStr() + cRefineLines_.timeStr();
}

void HoughLinesV::initialize(const cv::Size &size, int rowType, int accType, uint collectThreshold)
{
	accType_ = accType;
	rowType_ = rowType;
	maxLines_ = 1024;
    uint wx = 64, wy = 45, accH = 128;
    uint localMemorySize = (uint)set_->getLocalSize();
    //uint wx = localMemorySize / (accH * cvTypeSize(rowType)) - wy + 1;
	uint horGroups = size.width / wx;// (size.width + wx - 1) / wx;
	uint verGroups = size.height / wy;
    //wx = size.width / horGroups;

	int accW = (int)alignSize(wx + wy - 1);
	assert(accW * accH * cvTypeSize(rowType_) < localMemorySize);

	const vector<pair<string, int> > params = {
        {"WX", (int)wx},
        {"WY", (int)wy},
		{"ACC_W", accW},
        {"ACC_H", accH},
        {"WIDTH", size.width},
		{"HEIGHT", size.height},
		{"FULL_ACC_W", alignSize(size.width + size.height - 1)},
		{"MAX_LINES", maxLines_}
    };
    loadKernels("hough.cl", params);
    /*if (paramsOut) {
        paramsOut->clear();
        for (const auto &p : params)
            paramsOut->insert(p);
    }*/
	using namespace cl;
	scanGlobalSize_ = NDRange(horGroups * localSize_[0], verGroups);

    source_		= MatBuffer(set_, size, CV_8U, CL_MEM_READ_ONLY);
    accs_		= MatBuffer(set_, cv::Size(accW, accH * verGroups * horGroups), rowType_);
	accRows_	= MatBuffer(set_, cv::Size(size.width + (accW - wx), accH * verGroups), rowType_);
	linesCount_ = BufferT<uint>(set_);
	lines_		= BufferT<LineV>(set_, maxLines_);
	flags_		= BufferT<uint>(set_, horGroups * verGroups);

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
	kCollectLines_.setArg(1, collectThreshold);
	uint accStep = accumulator.size().width;
	kCollectLines_.setArg(2, accStep);
	kCollectLines_.setArg(3, linesCount_);
	kCollectLines_.setArg(4, lines_);

	// Kernel refineLines: source_, lines_ => lines_
	kRefineLines_.setArg(0, source_);
	kRefineLines_.setArg(1, lines_);
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
	//accRows_.fill();
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
	set_->queue.enqueueNDRangeKernel(kSumAccumulator_, cl::NDRange(), cl::NDRange(localSize_[0] * 8), localSize_, nullptr, &e);
	cSumAccumulator_.inc(e);
}

void HoughLinesV::readLines(std::vector<LineV>& lines)
{
	uint count;
	set_->queue.enqueueReadBuffer(linesCount_, true, 0, sizeof(uint), &count);
	if (count > maxLines_) {
		printf("Too much lines: %d\n", count);
		count = maxLines_;
	}
	lines.resize(count);
	if (count)
		set_->queue.enqueueReadBuffer(lines_, true, 0, count * sizeof(LineV), lines.data());
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

void HoughLinesV::refineLines()
{
	using namespace cl;
	uint count;
	set_->queue.enqueueReadBuffer(linesCount_, true, 0, sizeof(uint), &count);
	if (count > maxLines_) count = maxLines_;
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kRefineLines_, NDRange(), NDRange(localSize_[0] * count), localSize_, nullptr, &e);
	cRefineLines_.inc(e);
}

void HoughLinesV::refineLines(std::vector<LineV> &lines)
{
	if (lines.empty()) return;
	lines_.write(lines);
	cl::Event e;
	set_->queue.enqueueNDRangeKernel(kRefineLines_, cl::NDRange(), cl::NDRange(localSize_[0] * lines.size()), localSize_, nullptr, &e);
	cRefineLines_.inc(e);
	lines_.read(lines, lines.size());
}

void HoughLinesV::filterLines(std::vector<LineV>& lines)
{
	std::vector<LineV> result;
	int height = source_.size().height;
	for (const auto &s : lines) {
		bool found = false;
		for (auto &r : result)
            if (fabs(r.fb - s.fb) < 4) {
				float re = r.fb + r.fa * height;
				float se = s.fb + r.fa * height;
                if (fabs(re - se) < 4) {
					found = true;
					if (r.value < s.value)
						r = s;
				}
			}
		if (!found)
			result.push_back(s);
	}
	lines = result;
}

#include <opencv2/imgproc.hpp>


void HoughLinesV::drawMarkers(cv::Mat & out, const std::vector<LineV> &lines)
{
	if (out.type() == CV_8U)
		cv::cvtColor(out, out, CV_GRAY2RGB);
	cv::Mat markers = cv::Mat::zeros(out.size(), CV_8UC3);
	for (auto &l : lines) {
		cv::drawMarker(markers, cv::Point(l.b, l.a), cv::Scalar(255, 0, 0, 50), cv::MARKER_SQUARE);
	}
	cv::addWeighted(out, 1, markers, 0.2, 0, out);
}

cv::Mat HoughLinesV::drawLines(const cv::Mat &src, const std::vector<LineV> &lines)
{
	cv::Mat out;
	src.copyTo(out);
	if (out.type() == CV_8U)
		cv::cvtColor(out, out, CV_GRAY2RGB);
	if (out.type() == CV_8UC4)
		cv::cvtColor(out, out, CV_RGBA2RGB);
	uint m = 0;
	for (auto &l : lines)
		if (l.value > m)
			m = l.value;

	cv::Mat markers = out;// cv::Mat::zeros(out.size(), CV_8UC3);
	for (auto &l : lines) {
		/*int shift = l.a * (markers.rows - 1) / 127;
		int b = l.b - shift;
		cv::line(markers, cv::Point(b, 0), cv::Point(b + int((2.0 * l.a / 127. - 1.0) * markers.rows), markers.rows - 1), cv::Scalar(255 * l.value / m, 0, 0, 150));*/
		cv::line(markers, cv::Point((int)l.fb, 0), cv::Point(int(l.fb + l.fa * markers.rows), markers.rows - 1), cv::Scalar(255 * l.value / m, 0, 0, 150));
	}
	//cv::addWeighted(out, 1, markers, 0.9, 0, out);
	return out;
}
