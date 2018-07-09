#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

#include "gtest_utils.h"
#include "../kalmanline.h"
#include "../cameramodel.h"


TEST(KalmanLineTest, predict)
{
	KalmanLine<double> kl1(Vector2d(2.75, 5), Vector2d(0.5, 0.0), 0);
	KalmanLine<double> kl2(Vector2d(1.25, 7), Vector2d(0.5, 0.0), 0);
	Quaterniond r; r = Quaterniond::FromTwoVectors(Vector3d(0.5, 1, 0), Vector3d(0, 1, 0));
	Vector3d t(2, -1.5, 3);
	auto rm = r.inverse().toRotationMatrix();
	kl1.predict(rm, t);
	EXPECT_NEAR(kl1.position[0], 0, 1E-10);
	EXPECT_NEAR(kl1.position[1], 5 - 3, 1E-10);
	kl2.predict(rm, t);
	EXPECT_NEAR(kl2.position[0], -2 * sqrt(1.25), 1E-10);
	EXPECT_NEAR(kl2.position[1], 7 - 3, 1E-10);
}

TEST(KalmanLineTest, getF)
{
	const double dx = 1E-6;
	Vector4d x0(2.75, 5, 0.5, -0.3);
	Quaterniond r = Quaterniond::FromTwoVectors(Vector3d(0.5, 1.1, -0.1), Vector3d(0, 1, 0.2));
	auto irm = r.inverse().toRotationMatrix();
	Vector3d t(2, -1.5, 3);

	KalmanLine<double> kl0(x0, 0);
	auto d = irm * Vector3d(kl0.direction[0], 1, kl0.direction[1]);
	auto p = irm * (Vector3d(kl0.position[0], 0, kl0.position[1]) - t);
	kl0.predict(irm, t);
	auto x0p = kl0.getX();
	Matrix4d testF = Matrix4d::Zero(), approxF;
	kl0.getF(testF, irm, p, d);

	for (int i = 0; i < 4; i++) {
		double tmp = x0[i];
		x0[i] += dx;
		KalmanLine<double> kl1(x0, 0);
		x0[i] = tmp;
		kl1.predict(irm, t);
		auto x1p = kl1.getX();
		auto d = (x1p - x0p) / dx;
		approxF.block<4, 1>(0, i) = d;
	}
	//std::cout << "testF:" << std::endl << testF << std::endl;
	//std::cout << "approxF:" << std::endl << approxF << std::endl;
	EXPECT_TRUE(MatrixMatch(approxF, testF, 1E-5));
}

TEST(KalmanLineTest, correct)
{
	CameraModel<double> cm;
	cm.setParams(0.1, 500, 640, 500, 360, 500, 640, 500, 360);

	KalmanLine<double> kl1(Vector2d(2.75, 5), Vector2d(0.5, 0.0), 0);
	Quaterniond r; r = Quaterniond::FromTwoVectors(Vector3d(0.5, 1, 0), Vector3d(0, 1, 0));
	Vector3d t(2, -1.5, 3);
	auto rm = r.inverse().toRotationMatrix();
	LineV leftLine, rightLine;
	Matrix2d R = Matrix2d::Identity() * 0.1;
	kl1.correct<false>(cm, leftLine, R);
	kl1.correct<true>(cm, rightLine, R);

}
