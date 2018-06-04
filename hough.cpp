#define get_group_id(n) (groupId[n])
#define get_local_id(n) (localId[n])
#define get_group_size(n) 1
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


uint groupId[] = { 0, 2 };
uint localId[] = { 0 };
uint numGroups[] = {2, 3};

#include <assert.h>
#include <math.h>
#include "hough.cl"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void houghTest()
{
	cv::Mat src = cv::Mat::zeros(cv::Size(1280, 720), CV_8U);
	cv::line(src, cv::Point(200, 60), cv::Point(300, 700), cv::Scalar(1));
	cv::line(src, cv::Point(400, 60), cv::Point(350, 200), cv::Scalar(1));
	cv::line(src, cv::Point(600, 5), cv::Point(700, 200), cv::Scalar(1));

	cv::Mat accs = cv::Mat::zeros(cv::Size(ACC_W, ACC_H * numGroups[0] * numGroups[1]), CV_8U);

	numGroups[1] = 3;
	for (groupId[1] = 0; groupId[1] < numGroups[1]; groupId[1]++)
		for (groupId[0] = 0; groupId[0] < numGroups[0]; groupId[0]++)
			houghScan(src.data, 1280, (ACC_TYPE *)accs.data);

	numGroups[1] = 1;

	for (groupId[0] = 0; groupId[0] < numGroups[0]; groupId[0]++)


	double min, max;
	cv::minMaxLoc(accs, &min, &max);
	accs.convertTo(accs, CV_8U, 255 / max);
	cv::minMaxLoc(src, &min, &max);
	src.convertTo(src, CV_8U, 255 / max);
	cv::imshow("hough", accs);
	cv::imshow("src", src);
	cv::waitKey();
}
