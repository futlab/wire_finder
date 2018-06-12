#ifndef HOUGH_H
#define HOUGH_H

#include <map>
#include <limits>

// OpenCV
#include <opencv2/core.hpp>

// OpenCL
#include "cl_utils.h"


void houghTest(CLSet *set);

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
    CLSet *set_;
    cl::Kernel kAccumulate_, kCollectLines_;
    void loadKernels(const std::string &fileName, const std::vector<std::pair<std::string, int>> &params);
    cl::NDRange localSize_, scanGlobalSize_;
    cl::Buffer source_, accs_, acc_, lines_, linesCount_;
    cv::Size sourceSize_, accsSize_;
	int accType_;
public:
    void initialize(const cv::Size &size, int accType = CV_16U, std::map<std::string, int> *paramsOut = nullptr);
    HoughLinesV(CLSet *set);
    void find(const cv::Mat &source, cv::Mat &result);
    void accumulate(const cv::Mat &source);
	void readAccumulator(cv::Mat &result);
	void readLines(std::vector<LineV> &lines);
	void collectLines();
	void collectLines(const cv::Mat &acc);

	// Reference:
	template<typename ACC_TYPE = unsigned char, int ACC_H = 128>
	void accumulateRef(const cv::Mat &source, cv::Mat &acc) {
		// Declare and initialize accumulator
		const uint accW = source.cols + source.rows - 1;
		acc = cv::Mat::zeros(cv::Size(accW, ACC_H), cvType<ACC_TYPE>());

		// Shift parameters
		const uint shiftStep = (uint)(65536.f / (ACC_H - 1) * (source.rows - 1));
		//printf("shiftStart %d, shiftStep %d\n", shiftStart, shiftStep);

		// Scan image window
		const uchar *pSrc = (const uchar*)source.data;
		for (uint y = 0; y < (uint)source.rows; ++y) {
			uint bStep = (uint)(2.0f / (ACC_H - 1) * 65536.f * y);
			for (int xc = 0; xc < source.cols; ++xc) {
				uchar value = *(pSrc++);
				ACC_TYPE *pAcc = (ACC_TYPE *)acc.data;
				uint b = ((xc + y) << 16) | 0x8000;
				uint shift = 0;
				for (uint a = 0; a < ACC_H; a++) {
					//float bf = x + xc - y * ((float)a * 2.0f / (ACC_H - 1) - 1.0f);
					//assert(round(bf) == (signed(b) >> 16));
					uint idx = (b - shift & 0xFFFF0000) >> 16;
					//if (value) printf("xc: %d, a: %d; idx: %d\n", xc, a, idx);
					//assert(idx < ACC_W);
					pAcc[idx] = add_sat<ACC_TYPE>(pAcc[idx], value);
					b -= bStep;
					shift -= shiftStep;
					pAcc += accW;
				}
			}
		}
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
