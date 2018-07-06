#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

#include "../kalmanline.h"


TEST(KalmanLineTest, predict)
{
	KalmanLine<double> kl1(Vector2d(2.75, 5), Vector3d(0.5, 1.0, 0.0), 0);
	KalmanLine<double> kl2(Vector2d(1.25, 7), Vector3d(0.5, 1.0, 0.0), 0);
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

