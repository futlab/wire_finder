#define get_group_id(n) (groupId[n])
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
uint wx_, wy_, accH_ = 128;

#include <assert.h>
#include <iostream>
#include <math.h>
#include "hough.cl"
//#define TESTING
#include <map>
#include "hough.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cl_utils.h"

#define SHOW_RES

bool testScanOneGroup(CLSet *set)
{
    uint w = 320, h = 45;
    cv::Size size(w, h);
    cv::Mat src = cv::Mat::zeros(size, CV_8U);
    src.data[2 + w * 15] = 1;
    cv::line(src, cv::Point(200, 1), cv::Point(205, 37), cv::Scalar(1));
    cv::line(src, cv::Point(100, 21), cv::Point(90, 39), cv::Scalar(1));
    cv::line(src, cv::Point(300, 5), cv::Point(315, 10), cv::Scalar(1));

    HoughLinesV hlv(set);

    std::map<std::string, int> params;
    hlv.initialize(size, &params);
    WX = params.find("WX")->second;
    WY = params.find("WY")->second;
    ACC_H = params.find("ACC_H")->second;
    cv::Size accSize(ACC_W, ACC_H);
    cv::Mat accs = cv::Mat::zeros(accSize, CV_8U);
    cv::Mat accsCL = cv::Mat::zeros(accSize, CV_8U);

    groupId[0] = 0;
    groupId[1] = 0;
    accumulate(src.data, w, (ACC_TYPE *)accs.data);

    hlv.accumulate(src);
    hlv.readAccumulator(accsCL);

    uchar t1[20], t2[20];
    memcpy(t1, accs.data, 20);
    memcpy(t2, accsCL.data, 20);

    cv::Mat cmp;
    cv::compare(accs, accsCL, cmp, cv::CMP_NE);
    int result = cv::countNonZero(cmp);

    std::cout << "Hough test scan one group: " << (result ? std::to_string(result) + " errors" : "Ok") << std::endl;
#ifdef SHOW_RES
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    src.convertTo(src, CV_8U, 255 / max);
    cv::imshow("src", src);

    cv::minMaxLoc(accs, &min, &max);
    accs.convertTo(accs, CV_8U, 255 / max);
    cv::imshow("acc", accs);

    cv::minMaxLoc(accsCL, &min, &max);
    accsCL.convertTo(accsCL, CV_8U, 255 / max);
    cv::imshow("accCL", accsCL);

    if (result)
        cv::imshow("result", cmp);

    cv::waitKey();
#endif
    return !result;
}


void houghTest(CLSet *set)
{
    //testScanOneGroup(set);

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
