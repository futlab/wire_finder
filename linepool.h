#ifndef LINEPOOL_H
#define LINEPOOL_H

#include <vector>
#include <forward_list>
#include <memory>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "defs.h"
#include "cameramodel.h"
#include "kalmanline.h"

using namespace Eigen;


template<typename real>
class LinePool
{
private:
	struct Cell
	{
		KalmanLine<real> *line;
		uint value;
	};
	std::forward_list<KalmanLine<real>> lines;
	std::vector<Cell> lineMapData[2], *lineMap, *lineMapNew;
	size_t prevLeftCount, prevRightCount;

	static void findGoodPairs(const std::vector<uint> &cmp, const std::vector<LineV> & left, const std::vector<LineV> & right, std::function<void(int l, int r, uint value)> onPair);
	friend class LinePoolTest;
public:
	std::unique_ptr<KalmanLine<real>> testLine;
	typedef Matrix<real, 3, 1> Vector3;
	CameraModel<real> camera;
	void onLines(const std::vector<LineV> &leftLines, const std::vector<LineV> &rightLines, const std::vector<unsigned int> &compare, const std::vector<unsigned int> &leftCompare, const std::vector<unsigned int> &rightCompare) {}

	//void show(float leftHFov, float rightHFov) {}
	void drawMarker(cv::Mat &left, cv::Mat &right, const Vector3f &point) {}
	void draw(cv::Mat &left, cv::Mat &right) {}
	void predict(const Quaternion<real> & r, const Vector3 & t)
	{
		if (!testLine) return;
		Matrix<real, 3, 3> irm = r.inverse().toRotationMatrix();
		testLine->predict(irm, t, 0.2, 0.1);
		//for (auto & hl : lines)
		//	hl.predict(rm, t);
	}
	void correct(const LineV & left, const LineV & right)
	{
		if (!testLine) {
			testLine = std::make_unique<KalmanLine<real>>(camera, left, right, 1, 0.1);
		} 
		else {
			Matrix2f R;
			R << 0.01f, 0.f, 0.f, 3.f;
			testLine->correct(camera, left, right, R);
		}
	}
    LinePool() : 
		prevLeftCount(0), prevRightCount(0), lineMap(lineMapData), lineMapNew(lineMapData + 1)
	{

	}
    void show(float leftHFov, float rightHFov, const KalmanLine<real> *best /*, const std::pair<int, int> selection*/)
	{
		float maxZ = 20;
		float minX = -1, maxX = 1;
		for (auto & hl : lines) {
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
		float baseline = camera.getBaseline();
		cv::line(plot, cv::Point(dx, dz), cv::Point(dx + int(baseline * k), dz), cv::Scalar(255, 0, 0));
		const float cl = 150;
		int leftCos = int(cl * cosf(leftHFov / 2)), leftSin = int(cl * sinf(leftHFov / 2));
		int rightCos = int(cl * cosf(rightHFov / 2)), rightSin = int(cl * sinf(rightHFov / 2));
		cv::line(plot, cv::Point(dx, dz), cv::Point(dx + leftSin, dz + leftCos), cv::Scalar(255, 0, 0));
		cv::line(plot, cv::Point(dx, dz), cv::Point(dx - leftSin, dz + leftCos), cv::Scalar(255, 0, 0));
		cv::line(plot, cv::Point(dx + int(baseline * k), dz), cv::Point(dx + int(baseline * k) + rightSin, dz + rightCos), cv::Scalar(255, 0, 0));
		cv::line(plot, cv::Point(dx + int(baseline * k), dz), cv::Point(dx + int(baseline * k - rightSin), dz + rightCos), cv::Scalar(255, 0, 0));

		//for (auto & hl : lines) 
		if (testLine) {
			auto &hl = *testLine;
			int px = dx + int(k * hl.position[0]), py = dz + int(k * hl.position[1]);
			cv::circle(plot, cv::Point(px, py), 2, cv::Scalar(0, /*hl.p **/ 255, 0));
			int pxe = dx + int(k * (hl.position[0] + hl.direction[0])), pye = dz + int(k * (hl.position[1] + hl.direction[1]));
			cv::line(plot, cv::Point(px, py), cv::Point(pxe, pye), cv::Scalar(0, 255, 255));
		}

		
		if (best){
			auto &hl = *best;
			int px = dx + int(k * hl.position[0]), py = dz + int(k * hl.position[1]);
			cv::circle(plot, cv::Point(px, py), 2, cv::Scalar(0, /*hl.p **/ 255, 0));
			int pxe = dx + int(k * (hl.position[0] + hl.direction[0])), pye = dz + int(k * (hl.position[1] + hl.direction[1]));
			cv::line(plot, cv::Point(px, py), cv::Point(pxe, pye), cv::Scalar(0, 0, 255));
		}

		cv::imshow("Pool", plot);
	}

};

#endif // LINEPOOL_H
