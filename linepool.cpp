#include "linepool.h"

LinePool::LinePool()
{

}

void LinePool::setCameraParams(float baseline, float lfx, float lcx, float lfy, float lcy, float rfx, float rcx, float rfy, float rcy)
{
	this->baseline = baseline;
	this->lfx = lfx;
	this->lcx = lcx;
	this->lfy = lfy;
	this->lcy = lcy;
	this->rfx = rfx;
	this->rcx = rcx;
	this->rfy = rfy;
	this->rcy = rcy;
}

void LinePool::getHypoLine(HypoLine & out, const LineV & left, const LineV & right)
{
	// Coordinates X of the line in the pixels at the vertical center:
	float leftX  = left.b  + (float)left.a  / 32768.0f * lcy - lcx;
	float rightX = right.b + (float)right.a / 32768.0f * rcy - rcx;
	// Coordinates of line center in the left camera camera coordinate center
	float x = baseline * leftX * rfx / (leftX * rfx - rightX * lfx);
	float z = x * lfx / leftX;
	out.position << x, z;
}

void LinePool::onLines(const std::vector<LineV> & leftLines, const std::vector<LineV> & rightLines, const std::vector<unsigned int> &compare)
{
	pool_.clear();
	const unsigned int threshold = 1000000;
	auto it = compare.begin();
	for (auto & left : leftLines)
		for (auto & right : rightLines) {
			unsigned int c = *(it++) - 1;
			if (c < threshold) {
				HypoLine hl;
				getHypoLine(hl, left, right);
				hl.p = 1.0f - (float)c / threshold;
				pool_.push_back(hl);
			}
		}
}

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void LinePool::show(float leftHFov, float rightHFov/*, const std::pair<int, int> selection*/)
{
	float maxZ = 20;
	float minX = -1, maxX = 1;
	for (auto & hl : pool_) {
		if (hl.position[0] < minX)
			minX = hl.position[0];
		if (hl.position[0] > maxX)
			maxX = hl.position[0];
		if (hl.position[1] > maxZ)
			maxZ = hl.position[1];
	}
	if (-minX > maxX)
		maxX = -minX;
	else
		minX = -maxX;
	int h = 600, w = 800, dz = 20;
	float kz = (h - dz) / maxZ;
	float kx = w / (maxX - minX);
	float k = MIN(kx, kz);
	int dx = w / 2;
	cv::Mat plot = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
	cv::line(plot, cv::Point(dx, dz), cv::Point(dx + int(baseline * k), dz), cv::Scalar(255, 0, 0));
	const float cl = 150;
	int leftCos = int(cl * cosf(leftHFov / 2)), leftSin = int(cl * sinf(leftHFov / 2));
	int rightCos = int(cl * cosf(rightHFov / 2)), rightSin = int(cl * sinf(rightHFov / 2));
	cv::line(plot, cv::Point(dx, dz), cv::Point(dx + leftSin, dz + leftCos), cv::Scalar(255, 0, 0));
	cv::line(plot, cv::Point(dx, dz), cv::Point(dx - leftSin, dz + leftCos), cv::Scalar(255, 0, 0));
	cv::line(plot, cv::Point(dx + int(baseline * k), dz), cv::Point(dx + int(baseline * k) + rightSin, dz + rightCos), cv::Scalar(255, 0, 0));
	cv::line(plot, cv::Point(dx + int(baseline * k), dz), cv::Point(dx + int(baseline * k - rightSin), dz + rightCos), cv::Scalar(255, 0, 0));

	for (auto & hl : pool_) {
		int px = dx + int(k * hl.position[0]), py = dz + int(k * hl.position[1]);
		cv::circle(plot, cv::Point(px, py), 2, cv::Scalar(0, hl.p * 255, 0));
	}
	cv::imshow("Pool", plot);
}

