#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "linepool.h"

class LinePoolTest
{
public:
	void testLinePredict();

};

void LinePoolTest::testLinePredict()
{
	/*LinePool<double> lp;
	Vector2d position(2, 5);
	Vector3d direction(1, -2, 0);
	direction.normalize();
	lp.lines.emplace_front(position, direction, 1.0);

	Quaterniond r; r = AngleAxisd(M_PI / 6, Vector3d(0, 0, 1));
	Vector3d t(0, -1, 0);
	Vector3d v(2, 0, 5);
	Vector3d p = r.inverse().toRotationMatrix() * v - t;
	std::cout << p << std::endl;
	lp.camera.predict(r, t, v);
	//lp.predict(r, t);*/

	return;
}


void testLinepool()
{
	LinePoolTest lpt;
	lpt.testLinePredict();

}
