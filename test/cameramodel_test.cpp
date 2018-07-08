#include "../cameramodel.h"

#include <iostream>
#include <gtest/gtest.h>

#include "gtest_utils.h"


TEST(CameraModelTest, projectLine)
{
	CameraModel<double> cm;
	cm.setParams(0.1, 500, 640, 500, 360, 500, 640, 500, 360);
	for (int x = 0; x < 100; x++) {
		Vector2d pos(x % 5 - 2.5, 1 + x % 3), dir((x % 7) * 0.1 - 0.3, (x % 13) * 0.1 - 0.5);
		double al, bl, ar, br;
		cm.projectLine<false>(al, bl, pos, dir);
		cm.projectLine<true>(ar, br, pos, dir);
		Vector3d p, p1;
		cm.pointByScreen(p, bl, br, cm.leftCenterY());
		EXPECT_NEAR(p[0], pos[0], 1E-10);
		EXPECT_NEAR(p[1], 0, 1E-10);
		EXPECT_NEAR(p[2], pos[1], 1E-10);
		cm.pointByScreen(p1, bl + al * 100, br + ar * 100, cm.leftCenterY() + 100);
		auto d = p1 - p;
		EXPECT_NEAR(d[0] / d[1], dir[0], 1E-10);
		EXPECT_NEAR(d[2] / d[1], dir[1], 1E-10);
	}
}

TEST(CameraModelTest, projectLineDiff)
{
	const double dx = 1E-7;
	CameraModel<double> cm;
	cm.setParams(0.1, 500, 640, 500, 360, 500, 640, 500, 360);
	for (int i = 0; i < 100; i++) {
		Vector2d pos(i % 5 - 2.5, 1 + i % 3), dir((i % 7) * 0.1 - 0.3, (i % 13) * 0.1 - 0.5);
		Vector2d leftLine, rightLine;
		cm.projectLine<false>(leftLine, pos, dir);
		cm.projectLine<true>(rightLine, pos, dir);
		Vector4d x; x << pos, dir;
		Matrix<double, 2, 4> testHL = Matrix<double, 2, 4>::Zero(), testHR = Matrix<double, 2, 4>::Zero(), approxHL, approxHR;
		cm.projectLineDiff<false>(testHL, pos, dir);
		cm.projectLineDiff<true> (testHR, pos, dir);
		for (int j = 0; j < 4; j++) {
			Vector4d x1 = x;
			x1[j] += dx;
			pos = x1.block<2, 1>(0, 0);
			dir = x1.block<2, 1>(2, 0);
			Vector2d leftLine1, rightLine1;
			cm.projectLine<false>(leftLine1, pos, dir);
			cm.projectLine<true>(rightLine1, pos, dir);
			approxHL.block<2, 1>(0, j) = (leftLine1 - leftLine) / dx;
			approxHR.block<2, 1>(0, j) = (rightLine1 - rightLine) / dx;
		}
		//std::cout << "testHL:" << std::endl << testHL << std::endl;
		//std::cout << "approxHL:" << std::endl << approxHL << std::endl;
		EXPECT_TRUE(MatrixMatch(approxHL, testHL, 1E-3));
		EXPECT_TRUE(MatrixMatch(approxHR, testHR, 1E-3));
	}
}
