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
	cv::addWeighted(out, 1, markers, 0.1, 0, out);
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

bool testScanOneGroup(CLSet *set)
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
    hlv.readAccumulator(accsCL);

    cv::Mat cmp;
    cv::compare(accs, accsCL, cmp, cv::CMP_NE);
    int result = cv::countNonZero(cmp);

	std::cout << "Hough test scan one group: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;

	std::vector<LineV> lines, linesCL;
	hlv.collectLinesRef<ushort>(accs, 20, lines);
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

bool testScanOneRow(CLSet *set)
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

bool testScan(CLSet *set)
{
	uint w = 640, h = 180;
	cv::Size size(w, h);
	cv::Mat src = cv::Mat::zeros(size, CV_8U);
	src.data[2 + w * 15] = 1;
	for (int x = 0; x < 600; x += 50 + x / 4) {
		int y = (x & 0xF) * 10;
		cv::line(src, cv::Point(50 + x, 1 + y), cv::Point(55 + x, 37), cv::Scalar(1));
		cv::line(src, cv::Point(100 + x, 21), cv::Point(90 + x, 39 + y), cv::Scalar(1));
		cv::line(src, cv::Point(103 + x, 21 + y), cv::Point(88 + x, 39), cv::Scalar(1));
		cv::line(src, cv::Point(3 + x, 5), cv::Point(15 + x, 10 + y), cv::Scalar(1));
	}

	HoughLinesV hlv(set);

	hlv.initialize(size, CV_16U);
	cv::Mat acc, accs; //= cv::Mat::zeros(accSize, CV_8U);
	cv::Mat accCL, accsCL; //= cv::Mat::zeros(accSize, CV_8U);

	hlv.accumulateRef<ushort>(src, acc);

	for (int x = 0; x < 4; x++) {
		cv::Mat a, s = src(cv::Rect(0, x * 45, src.cols, 45));
		hlv.accumulateRef<ushort>(s, a);
		accs.push_back(a);
	}

	hlv.accumulateRows(src, accsCL);
	hlv.sumAccumulator(accCL);

	cv::Mat cmpRows;
	cv::compare(accs, accsCL, cmpRows, cv::CMP_NE);
	int resultRows = cv::countNonZero(cmpRows);
	std::cout << "Hough test scan rows: " << (resultRows ? std::to_string(resultRows) + " errors" : "Ok") << std::endl;

	cv::Mat cmpFull;
	cv::compare(acc, accCL, cmpFull, cv::CMP_NE);
	int resultFull = cv::countNonZero(cmpFull);
	std::cout << "Hough test scan full: " << (resultFull ? std::to_string(resultFull) + " errors" : "Ok") << std::endl;

	std::vector<LineV> lines, linesCL;
	hlv.collectLinesRef<ushort, 4>(acc, 10, lines);
	
	hlv.collectLines(accCL);
	hlv.readLines(linesCL);

	int result = compareLines(lines, linesCL);
	std::cout << "Compare lines: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;

#ifdef SHOW_RES
	showScaled("src", src);
	showScaled("acc", acc, lines);
	showScaled("accCL", acc, lines);
	showScaled("accs", accs);
	showScaled("accsCL", accsCL);
	if (resultFull)
		cv::imshow("result", cmpFull);
	if (resultRows)
		cv::imshow("result", cmpRows);
	cv::waitKey();
#endif
	return !result;
}

void houghTest(CLSet *set)
{
    testScan(set);

	cv::Mat src = cv::Mat::zeros(cv::Size(1280, 720), CV_8U);
	cv::line(src, cv::Point(200, 60), cv::Point(300, 700), cv::Scalar(1));
	cv::line(src, cv::Point(400, 60), cv::Point(350, 200), cv::Scalar(1));
	cv::line(src, cv::Point(600, 5), cv::Point(700, 200), cv::Scalar(1));

    cv::Mat accs;// = cv::Mat::zeros(cv::Size(ACC_W, ACC_H * numGroups[0] * numGroups[1]), CV_8U);

    HoughLinesV hlv(set);
    hlv.initialize(src.size());

    //hlv
    hlv.find(src, accs);

    /*numGroups[1] = 3;
	for (groupId[1] = 0; groupId[1] < numGroups[1]; groupId[1]++)
		for (groupId[0] = 0; groupId[0] < numGroups[0]; groupId[0]++)
			houghScan(src.data, 1280, (ACC_TYPE *)accs.data);

    numGroups[1] = 1;*/

    //for (groupId[0] = 0; groupId[0] < numGroups[0]; groupId[0]++)


	double min, max;
	cv::minMaxLoc(accs, &min, &max);
	accs.convertTo(accs, CV_8U, 255 / max);
	cv::minMaxLoc(src, &min, &max);
	src.convertTo(src, CV_8U, 255 / max);
	cv::imshow("hough", accs);
	cv::imshow("src", src);
	cv::waitKey();
}
