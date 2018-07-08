#include "../cameramodel.h"

#include <iostream>
#include <gtest/gtest.h>

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
