#ifndef HOUGH_H
#define HOUGH_H

#include <map>
#include <limits>

// OpenCV
#include <opencv2/core.hpp>

// OpenCL
#include "cl_utils.h"


void houghTest(cl::Set *set);

template<typename T> int cvType();
template<> inline constexpr int cvType<uchar>() { return CV_8U; }
template<> inline constexpr int cvType<ushort>() { return CV_16U; }
template<typename T> T add_sat(T a, T b)
{
	if (int(a) + int(b) > std::numeric_limits<T>::max())
		return std::numeric_limits<T>::max();
	return a + b;
}

struct LineV
{
	ushort value, desc;
	short a, b;
};

class HoughLinesV
{
private:
    cl::Set *set_;
    cl::Kernel kAccumulate_, kAccumulateRows_, kSumAccumulator_, kCollectLines_;
	cl::Counter cAccumulate_, cAccumulateRows_, cSumAccumulator_, cCollectLines_;
    void loadKernels(const std::string &fileName, const std::vector<std::pair<std::string, int>> &params);
    cl::NDRange localSize_, scanGlobalSize_;
	cl::BufferT<uint> flags_;
	cl::BufferT<LineV> lines_;
	cl::BufferT<uint> linesCount_;
	int accType_, rowType_;
	uint bytesAlign_, flagsSize_, maxLines_;
	uint alignSize(uint size);
public:
	static void drawMarkers(cv::Mat & out, const std::vector<LineV> &lines);
	static cv::Mat drawLines(const cv::Mat &src, const std::vector<LineV> &lines);
	std::string getCounters();
	cl::MatBuffer source_, accs_, accumulator, accRows_;
	void initialize(const cv::Size &size, int rowType = CV_8U, int accType = CV_16U, uint collectThreshold = 10);
    HoughLinesV(cl::Set *set);
    void find(const cv::Mat &source, cv::Mat &result);
    void accumulate(const cv::Mat &source);
	void accumulateRows(const cv::Mat &source, cv::Mat &rows);
	void accumulateRows(const cv::Mat &source);
	void sumAccumulator();
	void readLines(std::vector<LineV> &lines);
	void collectLines();
	void collectLines(const cv::Mat &source);

	// Reference:
	template<typename ACC_TYPE = unsigned char, int ACC_H = 128>
	void accumulateRef(const cv::Mat &source, cv::Mat &acc, uint yw = 0) {
		// Declare and initialize accumulator
		const uint accW = alignSize(source.cols + source.rows - 1);

		acc = cv::Mat::zeros(cv::Size(accW, ACC_H), cvType<ACC_TYPE>());

		// Shift parameters
		const uint
			shiftStart = ((0 + yw) << 16) - yw * (65536 / ACC_H) + 0x8000,
			shiftStep = (2 * yw + source.rows - 1) * (65536 / ACC_H);
			//(uint)(65536.f / (ACC_H - 1) * (source.rows - 1));
		//printf("shiftStart %d, shiftStep %d\n", shiftStart, shiftStep);

		// Scan image window
		const uchar *pSrc = (const uchar*)source.data;
		for (uint y = 0; y < (uint)source.rows; ++y) {
			uint bStep = (y + yw) * (2 * 65536 / ACC_H);
			for (int xc = 0; xc < source.cols; ++xc) {
				uchar value = *(pSrc++);
				ACC_TYPE *pAcc = (ACC_TYPE *)acc.data;
				uint b = (((xc + y + yw) << 16) | 0x8000) - (y + yw) * (65536 / ACC_H);
				uint shift = shiftStart;
				for (uint a = 0; a < ACC_H; a++) {
					//float bf = x + xc - y * ((float)a * 2.0f / (ACC_H - 1) - 1.0f);
					//assert(round(bf) == (signed(b) >> 16));
					uint idx = (b - (shift & 0xFFFF0000)) >> 16;
					//if (value) printf("xc: %d, a: %d; idx: %d\n", xc, a, idx);
					assert(idx < (uint)(source.cols + source.rows - 1));
					pAcc[idx] = add_sat<ACC_TYPE>(pAcc[idx], value);
					b -= bStep;
					shift -= shiftStep;
					pAcc += accW;
				}
				assert(b + bStep == (((xc - y - yw) << 16) | 0x8000) + (y + yw) * (65536 / ACC_H));
			}
		}
	}
	template<typename ACC_TYPE = unsigned char>
	cv::Mat rectifyAccumulatorRef(const cv::Mat &acc, uint imageHeight)
	{
		const uint accH = acc.rows;
		const uint shiftStep = (imageHeight - 1) * (65536 / accH);
		uint shift = 0x8000;
		cv::Mat result = cv::Mat::zeros(cv::Size(acc.cols + imageHeight, accH), cvType<ACC_TYPE>());
		const ACC_TYPE *s = (const ACC_TYPE *)acc.data;
		ACC_TYPE *d = (ACC_TYPE *)result.data;
		for (uint a = 0; a < accH; ++a) {
			for (uint b = 0; b < (uint)acc.cols; b++)
				d[b + imageHeight + short(shift >> 16)] = *(s++);
			d += result.cols;
			shift -= shiftStep;
		}
		return result;
	}
	template<typename ACC_TYPE = unsigned char, uint D = 2>
	void collectLinesRef(const cv::Mat &acc, ACC_TYPE threshold, std::vector<LineV> &lines) {
		assert(acc.type() == cvType<ACC_TYPE>());
		const ACC_TYPE *pAcc = (const ACC_TYPE *)acc.data;
		const int winStep = acc.cols - 1 - 2 * D;
		for (uint yw = D; yw < acc.rows - D; ++yw)
			for (uint xw = D; xw < acc.cols - D; ++xw)
			{
				const ACC_TYPE value = pAcc[xw + yw * acc.cols];
				if (value < threshold)
					continue;
				const ACC_TYPE *pWin = pAcc + (xw - D + (yw - D) * acc.cols);
				for (uint y = yw - D; y <= yw + D; ++y) {
					for (uint x = xw - D; x <= xw + D; ++x, ++pWin) {
						if (value < *pWin) {
							pWin = nullptr;
							break;
						}
					}
					if (!pWin)
						break;
					pWin += winStep;
				}
				if (pWin) {
					lines.push_back({value, 0, (short)yw, (short)xw});
				}
			}
	}
};

#endif
