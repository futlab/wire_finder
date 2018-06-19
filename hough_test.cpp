/*#define get_group_id(n) (groupId[n])
#define get_local_id(n) (localId[n])
#define get_local_size(n) 1
#define get_num_groups(n) (numGroups[n])
#define __kernel
#define __global
#define __local
#define __private
#define mad24(a, b, c) ((a) * (b) + (c))
#define add_sat(a, b) ((a) + (b))
#define barrier(n) (0)
#define ACC_TYPE uchar
#define GS 1
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

uint groupId[] = { 0, 2 };
uint localId[] = { 0 };
uint numGroups[] = {2, 3};

#define WX wx_
#define WY wy_
#define ACC_H accH_
uint wx_, wy_, accH_ = 128;*/

#include <assert.h>
#include <iostream>
#include <math.h>
//#include "hough.cl"
//#define TESTING
#include <map>
#include "hough.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cl_utils.h"

#define SHOW_RES

using namespace cl;

#ifdef SHOW_RES
void showScaled(const cv::String &name, const cv::Mat src)
{
	double min, max;
	cv::minMaxLoc(src, &min, &max);
	cv::Mat out;
	src.convertTo(out, CV_8U, 255 / max);
	cv::imshow(name, out);
}

void showScaled(const cv::String &name, const cv::Mat src, const std::vector<LineV> &lines)
{
	double min, max;
	cv::minMaxLoc(src, &min, &max);
	cv::Mat out, markers = cv::Mat::zeros(src.size(), CV_8UC3);
	src.convertTo(out, CV_8U, 255 / max);
	cv::cvtColor(out, out, CV_GRAY2RGB);
	for (auto &l : lines) {
		cv::drawMarker(markers, cv::Point(l.b, l.a), cv::Scalar(255, 0, 0, 50), cv::MARKER_SQUARE);
	}
	cv::addWeighted(out, 1, markers, 0.2, 0, out);
	cv::imshow(name, out);
}

void showScaledDrawLines(const cv::String &name, const cv::Mat src, const std::vector<LineV> &lines)
{
	double min, max;
	cv::minMaxLoc(src, &min, &max);
	cv::Mat out, markers = cv::Mat::zeros(src.size(), CV_8UC3);
	src.convertTo(out, CV_8U, 255 / max);
	cv::cvtColor(out, out, CV_GRAY2RGB);
	uint m = 0;
	for (auto &l : lines)
		if (l.value > m)
			m = l.value;

	for (auto &l : lines) {
		/*int shift = l.a * (markers.rows - 1) / 127;
		int b = l.b - shift;
		cv::line(markers, cv::Point(b, 0), cv::Point(b + int((2.0 * l.a / 127. - 1.0) * markers.rows), markers.rows - 1), cv::Scalar(255 * l.value / m, 0, 0, 150));*/
		cv::line(markers, cv::Point(l.b, 0), cv::Point(l.b + (((int(l.a) * markers.rows) >> 15)), markers.rows - 1), cv::Scalar(255 * l.value / m, 0, 0, 150));

	}
	cv::addWeighted(out, 1, markers, 0.6, 0, out);
	cv::imshow(name, out);
}

#endif

int compareLines(const std::vector<LineV> &va, const std::vector<LineV> &vb)
{
	int res = int(vb.size() - va.size());
	for (auto &a : va) {
		bool found = false;
		for (auto &b : vb)
			if (a.a == b.a && a.b == b.b && a.value == b.value) {
				found = true;
				break;
			}
		if (!found) res++;
	}
	return res;
}

bool testScanOneGroup(Set *set)
{
    uint w = 160, h = 45;
    cv::Size size(w, h);
    cv::Mat src = cv::Mat::zeros(size, CV_8U);
    src.data[2 + w * 15] = 1;
    cv::line(src, cv::Point(50, 1), cv::Point(55, 37), cv::Scalar(1));
    cv::line(src, cv::Point(100, 21), cv::Point(90, 39), cv::Scalar(1));
	cv::line(src, cv::Point(103, 21), cv::Point(88, 39), cv::Scalar(1));
	cv::line(src, cv::Point(3, 5), cv::Point(15, 10), cv::Scalar(1));

    HoughLinesV hlv(set);

    hlv.initialize(size, CV_16U);
	cv::Mat accs; //= cv::Mat::zeros(accSize, CV_8U);
	cv::Mat accsCL; //= cv::Mat::zeros(accSize, CV_8U);

	hlv.accumulateRef<ushort>(src, accs);

    hlv.accumulate(src);
    hlv.accs_.read(accsCL);

    cv::Mat cmp;
    cv::compare(accs, accsCL, cmp, cv::CMP_NE);
    int result = cv::countNonZero(cmp);

	std::cout << "Hough test scan one group: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;

	std::vector<LineV> lines, linesCL;
	hlv.collectLinesRef<ushort>(accs, 20, lines, h);
	hlv.collectLines(accsCL);
	hlv.readLines(linesCL);

	result = compareLines(lines, linesCL);
	std::cout << "Compare lines: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;

#ifdef SHOW_RES
    showScaled("src", src);
    showScaled("acc", accs, lines);
    showScaled("accCL", accsCL, linesCL);
    if (result)
        cv::imshow("result", cmp);
    cv::waitKey();
#endif
    return !result;
}

bool testScanOneRow(Set *set)
{
	uint w = 640, h = 45;
	cv::Size size(w, h);
	cv::Mat src = cv::Mat::zeros(size, CV_8U);
	src.data[2 + w * 15] = 1;
	for (int x = 0; x < 600; x += 70 + x / 4) {
		cv::line(src, cv::Point(50 + x, 1), cv::Point(55 + x, 37), cv::Scalar(1));
		cv::line(src, cv::Point(100 + x, 21), cv::Point(90 + x, 39), cv::Scalar(1));
		cv::line(src, cv::Point(103 + x, 21), cv::Point(88 + x, 39), cv::Scalar(1));
		cv::line(src, cv::Point(3 + x, 5), cv::Point(15 + x, 10), cv::Scalar(1));
	}

	HoughLinesV hlv(set);

	hlv.initialize(size, CV_16U);
	cv::Mat accs; //= cv::Mat::zeros(accSize, CV_8U);
	cv::Mat accsCL; //= cv::Mat::zeros(accSize, CV_8U);

	hlv.accumulateRef<ushort>(src, accs);

	hlv.accumulateRows(src, accsCL);

	cv::Mat cmp;
	cv::compare(accs, accsCL, cmp, cv::CMP_NE);
	int result = cv::countNonZero(cmp);

	std::cout << "Hough test scan one row: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;

	std::vector<LineV> lines, linesCL;
	/*hlv.collectLinesRef<ushort>(accs, 10, lines);
	hlv.collectLines(accsCL);
	hlv.readLines(linesCL);*/

	result = compareLines(lines, linesCL);
	std::cout << "Compare lines: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;

#ifdef SHOW_RES
	showScaled("src", src);
	showScaled("acc", accs, lines);
	showScaled("accCL", accsCL, linesCL);
	if (result)
		cv::imshow("result", cmp);
	cv::waitKey();
#endif
	return !result;
}

bool testScan(Set *set)
{
	HoughLinesV hlv(set);

	uint w = 640, h = 180;
	cv::Size size(w, h);
	cv::Mat src = cv::Mat::zeros(size, CV_8U);
	src.data[1 + w * 0] = 1;
	for (int x = 0; x < 600; x += 50 + x / 4) {
		int y = (x & 0xF) * 10;
		cv::line(src, cv::Point(50 + x, 1 + y), cv::Point(55 + x, 37), cv::Scalar(1));
		cv::line(src, cv::Point(100 + x, 21), cv::Point(90 + x, 39 + y), cv::Scalar(1));
		cv::line(src, cv::Point(103 + x, 21 + y), cv::Point(88 + x, 39), cv::Scalar(1));
		cv::line(src, cv::Point(3 + x, 5), cv::Point(15 + x, 10 + y), cv::Scalar(1));
	}

	hlv.initialize(size, CV_16U, CV_16U);

	cv::Mat acc, accs; //= cv::Mat::zeros(accSize, CV_8U);
	cv::Mat accCL, accsCL; //= cv::Mat::zeros(accSize, CV_8U);

	hlv.accumulateRef<ushort>(src, acc);
	cv::Mat accRect = hlv.rectifyAccumulatorRef<ushort>(acc, h);

	for (int x = 0, yw = 0; x < 4; x++, yw += 45) {
		cv::Mat a, s = src(cv::Rect(0, x * 45, src.cols, 45));
		hlv.accumulateRef<ushort>(s, a, yw);
		accs.push_back(a);
	}

	for (int it = 0; it < 4; it++) {
		hlv.accumulateRows(src, accsCL);
		hlv.sumAccumulator();
		hlv.accumulator.read(accCL);

		cv::Mat cmpRows;
		//int tt = accsCL.type(), tt1 = accs.type();
		cv::compare(accs, accsCL, cmpRows, cv::CMP_NE);
		int resultRows = cv::countNonZero(cmpRows);
		std::cout << "Hough test scan rows: " << (resultRows ? std::to_string(resultRows) + " errors" : "Ok") << std::endl;

		cv::Mat cmpFull;
		cv::compare(acc, accCL, cmpFull, cv::CMP_NE);
		int resultFull = cv::countNonZero(cmpFull);
		std::cout << "Hough test scan full: " << (resultFull ? std::to_string(resultFull) + " errors" : "Ok") << std::endl;

		std::vector<LineV> lines, linesCL;
		int resultLines = 0;
		if (!resultFull && !resultRows) {
			hlv.collectLinesRef<ushort, 4>(accRect, 20, lines, h);

			hlv.collectLines(accCL);
			hlv.readLines(linesCL);

			resultLines = compareLines(lines, linesCL);
			std::cout << "Compare lines: " << (resultLines ? std::to_string(resultLines) + " errors" : "Ok") << std::endl;
		}

#ifdef SHOW_RES
		showScaledDrawLines("src", src, lines);
		showScaled("acc", acc);
		showScaled("accRect", accRect, lines);
		showScaled("accCL", accCL, lines);
		showScaled("accs", accs);
		showScaled("accsCL", accsCL);
		if (resultFull)
			cv::imshow("result", cmpFull);
		if (resultRows)
			cv::imshow("result", cmpRows);
		cv::waitKey();
#endif
	}
	return 0;
}

void houghTest(Set *set)
{
    testScan(set);

	cv::Mat src = cv::Mat::zeros(cv::Size(1280, 720), CV_8U);
	cv::line(src, cv::Point(200, 60), cv::Point(300, 700), cv::Scalar(1));
	cv::line(src, cv::Point(400, 60), cv::Point(350, 200), cv::Scalar(1));
	cv::line(src, cv::Point(600, 5), cv::Point(700, 200), cv::Scalar(1));

    cv::Mat accs;// = cv::Mat::zeros(cv::Size(ACC_W, ACC_H * numGroups[0] * numGroups[1]), CV_8U);

    HoughLinesV hlv(set);
    hlv.initialize(src.size());

    hlv.find(src, accs);

	double min, max;
	cv::minMaxLoc(accs, &min, &max);
	accs.convertTo(accs, CV_8U, 255 / max);
	cv::minMaxLoc(src, &min, &max);
	src.convertTo(src, CV_8U, 255 / max);
	cv::imshow("hough", accs);
	cv::imshow("src", src);
	cv::waitKey();
}


